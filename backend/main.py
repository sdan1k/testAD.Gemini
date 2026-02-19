"""
FastAPI сервер для гибридного поиска по решениям ФАС.
Запуск: uvicorn main:app --reload --port 8000

Использует Google Gemini Embedding API (новый SDK google-genai).
При недоступности Gemini - использует локальные эмбеддинги.

Поддержка развертывания:
- При наличии переменной DATA_DIR - ищет файлы там (для Render)
- Иначе использует локальную папку data/
- При первом запуске копирует файлы из репозитория в DATA_DIR если нужно
"""

import json
import re
import shutil
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Новый SDK google-genai
from google import genai
from google.genai import types

from config import Config
from industry_mapping import INDUSTRY_HIERARCHY, expand_filter_categories

# Конфигурация
BASE_DIR = Path(__file__).parent

# Используем Config.get_data_dir() для поддержки DATA_DIR
DATA_DIR = Config.get_data_dir()

# Пути к файлам эмбеддингов (по новой архитектуре)
EMBEDDINGS_FAS_ARGS_PATH = DATA_DIR / "embeddings_FAS_arguments.npy"
EMBEDDINGS_VIOLATION_PATH = DATA_DIR / "embeddings_violation_summary.npy"
EMBEDDINGS_AD_DESC_PATH = DATA_DIR / "embeddings_ad_description.npy"
CASES_PATH = DATA_DIR / "cases.json"

# Путь к исходным файлам в репозитории (для копирования при первом запуске)
REPO_DATA_DIR = BASE_DIR / "data"

# Модель Gemini Embedding 001
MODEL_NAME = "gemini-embedding-001"
# Размерность из конфига
EMBEDDING_DIMENSION = Config.EMBEDDING_DIMENSION

# Веса для полей при переранжировании
# FAS_arguments уже использован для первичного поиска!
FIELD_WEIGHTS = {
    'violation_summary': 0.6,
    'ad_description': 0.4,
}

# Количество кандидатов для первичного отбора
SEARCH_TOP_CANDIDATES = 100

# Иерархия регионов по федеральным округам
REGION_HIERARCHY = {
    "1. Центральный федеральный округ": [
        "Белгородская область", "Брянская область", "Владимирская область",
        "Воронежская область", "Ивановская область", "Калужская область",
        "Костромская область", "Курская область", "Липецкая область",
        "Московская область", "Орловская область", "Рязанская область",
        "Смоленская область", "Тамбовская область", "Тверская область",
        "Тульская область", "Ярославская область", "Город федерального значения Москва"
    ],
    "2. Южный федеральный округ": [
        "Республика Адыгея", "Республика Калмыкия", "Республика Крым",
        "Краснодарский край", "Астраханская область", "Волгоградская область",
        "Ростовская область", "Город федерального значения Севастополь"
    ],
    "3. Северо-Западный федеральный округ": [
        "Республика Карелия", "Республика Коми", "Архангельская область",
        "Вологодская область", "Калининградская область", "Ленинградская область",
        "Мурманская область", "Новгородская область", "Псковская область",
        "Ненецкий автономный округ", "Город федерального значения Санкт-Петербург"
    ],
    "4. Дальневосточный федеральный округ": [
        "Республика Саха (Якутия)", "Камчатский край", "Приморский край",
        "Хабаровский край", "Амурская область", "Магаданская область",
        "Сахалинская область", "Еврейская автономная область", "Чукотский автономный округ"
    ],
    "5. Сибирский федеральный округ": [
        "Республика Алтай", "Республика Бурятия", "Республика Тыва",
        "Республика Хакасия", "Алтайский край", "Забайкальский край",
        "Красноярский край", "Иркутская область", "Кемеровская область",
        "Новосибирская область", "Омская область", "Томская область"
    ],
    "6. Уральский федеральный округ": [
        "Курганская область", "Свердловская область", "Тюменская область",
        "Челябинская область", "Ханты-Мансийский автономный округ — Югра",
        "Ямало-Ненецкий автономный округ"
    ],
    "7. Приволжский федеральный округ": [
        "Республика Башкортостан", "Республика Марий Эл", "Республика Мордовия",
        "Республика Татарстан", "Удмуртская Республика", "Чувашская Республика",
        "Кировская область", "Нижегородская область", "Оренбургская область",
        "Пензенская область", "Ульяновская область", "Самарская область",
        "Саратовская область", "Пермский край"
    ],
    "8. Северо-Кавказский федеральный округ": [
        "Республика Дагестан", "Республика Ингушетия",
        "Кабардино-Балкарская Республика", "Карачаево-Черкесская Республика",
        "Республика Северная Осетия — Алания", "Чеченская Республика",
        "Ставропольский край"
    ]
}

def map_region_to_federal_district(region: str) -> str:
    """Определить федеральный округ по региону."""
    region_clean = region.strip()
    for district, regions in REGION_HIERARCHY.items():
        if region_clean in regions:
            return district
    return "Другие регионы"


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Нормализация оценки в диапазон 0-1."""
    import math
    # Защита от NaN и Inf
    if math.isnan(score) or math.isinf(score):
        return 0.0
    if score < min_val:
        return 0.0
    if score > max_val:
        return 1.0
    return (score - min_val) / (max_val - min_val)


def get_embedding(text: str, task_type: str = "retrieval_query") -> Optional[np.ndarray]:
    """
    Создать эмбеддинг для текста через Gemini API.
    При ошибке возвращает None.
    """
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIMENSION)
    
    # task_type это просто строка в новом SDK
    try:
        result = Config._genai_client.models.embed_content(
            model=MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSION
            )
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            print(f"⚠️ Квота Gemini API исчерпана!")
        elif "location" in error_msg.lower() or "not supported" in error_msg.lower():
            print(f"⚠️ Gemini API недоступен в вашем регионе!")
        else:
            print(f"Ошибка Gemini: {error_msg[:80]}...")
        return None


# Глобальные переменные
api_configured: bool = False
use_gemini: bool = True  # Флаг - использовать Gemini или локальные эмбеддинги

# Эмбеддинги для полей
embeddings_fas_args: Optional[np.ndarray] = None
embeddings_violation: Optional[np.ndarray] = None
embeddings_ad_desc: Optional[np.ndarray] = None

cases: Optional[list[dict]] = None


# Pydantic модели
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Поисковый запрос")
    top_k: int = Field(default=20, ge=1, le=50, description="Количество результатов")
    year: Optional[List[int]] = Field(default=None, description="Фильтр по году")
    region: Optional[List[str]] = Field(default=None, description="Фильтр по региону")
    industry: Optional[List[str]] = Field(default=None, description="Фильтр по отрасли")
    article: Optional[List[str]] = Field(default=None, description="Фильтр по статье закона")


class CaseResult(BaseModel):
    index: int
    score: float
    field_scores: Optional[Dict[str, float]] = None
    docId: Optional[str] = None
    Violation_Type: Optional[str] = None
    document_date: Optional[str] = None
    FASbd_link: Optional[str] = None
    FAS_division: Optional[str] = None
    violation_found: Optional[str] = None
    defendant_name: Optional[str] = None
    defendant_industry: Optional[str] = None
    ad_description: Optional[str] = None
    ad_content_cited: Optional[str] = None
    ad_platform: Optional[str] = None
    violation_summary: Optional[str] = None
    FAS_arguments: Optional[str] = None
    legal_provisions: Optional[str] = None
    thematic_tags: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    total_cases: int
    results: List[CaseResult]
    filters_applied: Optional[dict] = None
    message: Optional[str] = None


class SubIndustry(BaseModel):
    """Подотрасль"""
    name: str
    count: int
    sub_industries: List['SubIndustry'] = []


class IndustryGroup(BaseModel):
    """Группа отрасли с подотраслями"""
    name: str
    count: int
    sub_industries: List[SubIndustry] = []


class RegionGroup(BaseModel):
    """Группа региона по федеральному округу"""
    name: str
    count: int
    regions: List[str] = []

class ArticlePart(BaseModel):
    """Часть статьи"""
    name: str
    count: int


class ArticleGroup(BaseModel):
    """Группа статьи с частями"""
    name: str
    count: int
    parts: List[ArticlePart] = []


class FilterOptions(BaseModel):
    years: List[int]
    regions: List[str]
    region_groups: List[RegionGroup] = []  # Иерархия регионов по ФО
    industries: List[str]  # Плоский список отраслей
    industry_groups: List[IndustryGroup] = []  # Иерархия отраслей
    articles: List[str]
    article_groups: List[ArticleGroup] = []  # Иерархия статей


# FastAPI приложение
app = FastAPI(
    title="FAS Hybrid Search API",
    description="API для гибридного поиска по решениям ФАС о нарушениях в рекламе",
    version="4.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модифицируем Config для хранения клиента
Config._genai_client = None


def configure_gemini():
    """Настройка Gemini API."""
    global api_configured
    api_key = Config.get_api_key()
    Config._genai_client = genai.Client(api_key=api_key)
    api_configured = True
    print(f"Gemini API настроен с ключом: {api_key[:10]}...")


def init_data_dir():
    """
    Инициализация директории данных.
    Если DATA_DIR отличается от REPO_DATA_DIR и файлов там нет - копирует из репозитория.
    """
    # Проверяем, нужно ли копирование
    if DATA_DIR == REPO_DATA_DIR:
        return  # Используем локальную папку - всё ОК
    
    # Создаем директорию DATA_DIR если не существует
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Проверяем, есть ли уже файлы в DATA_DIR
    required_files = [
        "embeddings_FAS_arguments.npy",
        "embeddings_violation_summary.npy", 
        "embeddings_ad_description.npy",
        "cases.json"
    ]
    
    all_files_exist = all((DATA_DIR / f).exists() for f in required_files)
    
    if all_files_exist:
        print(f"Данные уже есть в {DATA_DIR}")
        return
    
    # Копируем файлы из репозитория
    print(f"Инициализация данных: копирование из {REPO_DATA_DIR} в {DATA_DIR}")
    for filename in required_files:
        src = REPO_DATA_DIR / filename
        dst = DATA_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Скопирован: {filename}")
        else:
            print(f"  ВНИМАНИЕ: Файл {filename} не найден в репозитории!")


def load_data():
    """Загрузка всех данных при старте."""
    global embeddings_fas_args, embeddings_violation, embeddings_ad_desc, cases, use_gemini
    
    print("=" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)
    
    # Инициализируем директорию данных (копируем файлы если нужно)
    init_data_dir()
    
    print(f"Директория данных: {DATA_DIR}")
    
    # Пробуем настроить Gemini API
    try:
        configure_gemini()
    except Exception as e:
        print(f"Не удалось настроить Gemini: {e}")
        use_gemini = False
    
    # Эмбеддинги FAS_arguments (для первичного поиска)
    if EMBEDDINGS_FAS_ARGS_PATH.exists():
        embeddings_fas_args = np.load(EMBEDDINGS_FAS_ARGS_PATH)
        print(f"  FAS_arguments эмбеддинги (первичный поиск): {embeddings_fas_args.shape}")
    else:
        print(f"  ВНИМАНИЕ: Файл {EMBEDDINGS_FAS_ARGS_PATH} не найден!")
    
    # Эмбеддинги violation_summary (для переранжирования)
    if EMBEDDINGS_VIOLATION_PATH.exists():
        embeddings_violation = np.load(EMBEDDINGS_VIOLATION_PATH)
        print(f"  violation_summary эмбеддинги: {embeddings_violation.shape}")
    
    # Эмбеддинги ad_description (для переранжирования)
    if EMBEDDINGS_AD_DESC_PATH.exists():
        embeddings_ad_desc = np.load(EMBEDDINGS_AD_DESC_PATH)
        print(f"  ad_description эмбеддинги: {embeddings_ad_desc.shape}")
    
    # Загрузка кейсов
    if CASES_PATH.exists():
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        print(f"  Кейсов загружено: {len(cases)}")
    else:
        print(f"  ВНИМАНИЕ: Файл {CASES_PATH} не найден!")
    
    print("=" * 50)
    if use_gemini:
        print(f"Режим: Gemini API (gemini-embedding-001)")
    else:
        print(f"Режим: Локальные эмбеддинги")
    print(f"Размерность: {EMBEDDING_DIMENSION}")
    print("=" * 50)


@app.on_event("startup")
async def startup_event():
    """Загрузка данных при старте."""
    try:
        load_data()
    except Exception as e:
        print(f"ОШИБКА ЗАГРУЗКИ: {e}")
        raise


def apply_filters(candidates: List[tuple], filters: dict) -> List[tuple]:
    """Применение фильтров к результатам поиска."""
    if not filters or not cases:
        return candidates
    
    filtered = []
    for idx, score in candidates:
        case = cases[idx]
        
        if filters.get('year'):
            if not case.get('document_date'):
                continue
            try:
                case_year = int(case['document_date'][:4])
                if case_year not in filters['year']:
                    continue
            except:
                continue
        
        if filters.get('region'):
            if not case.get('FAS_division') or case['FAS_division'] not in filters['region']:
                continue
        
        if filters.get('industry'):
            # Разворачиваем выбранные категории фильтра в реальные значения из БД
            expanded_industries = expand_filter_categories(filters['industry'])
            if not case.get('defendant_industry') or case['defendant_industry'] not in expanded_industries:
                continue
        
        if filters.get('article'):
            if not case.get('legal_provisions'):
                continue
            legal_raw = case['legal_provisions']
            
            # Парсим JSON массив если это строка типа "['п. 1 ч. 2 ст. 5', ...]"
            legal_provisions = []
            if isinstance(legal_raw, str):
                try:
                    # Пробуем распарсить как JSON массив
                    legal_provisions = json.loads(legal_raw.replace("'", '"'))
                except:
                    # Если не парсится, используем как есть
                    legal_provisions = [legal_raw]
            elif isinstance(legal_raw, list):
                legal_provisions = legal_raw
            else:
                legal_provisions = [str(legal_raw)]
            
            # Проверяем, есть ли в legal_provisions хотя бы один выбранный фильтр
            found = False
            for art in filters['article']:
                # Нормализуем фильтр для поиска - убираем точки для гибкого поиска
                # "ч. 3 ст. 5" -> "ч 3 ст 5"
                art_normalized = art.lower().replace('.', ' ').replace('  ', ' ').strip()
                
                # Извлекаем компоненты фильтра
                # Форматы: "ст. 5", "ч. 3 ст. 5", "ч.3 ст.5"
                filter_st_num = None
                filter_ch_num = None
                filter_p_num = None
                
                # Ищем "п. X ч. Y ст. Z"
                p_ch_st_match = re.search(r'п\s*(\d+)\s*ч\s*(\d+)\s*ст\s*(\d+)', art_normalized)
                if p_ch_st_match:
                    filter_p_num = p_ch_st_match.group(1)
                    filter_ch_num = p_ch_st_match.group(2)
                    filter_st_num = p_ch_st_match.group(3)
                else:
                    # Ищем "ч. Y ст. Z"
                    ch_st_match = re.search(r'ч\s*(\d+)\s*ст\s*(\d+)', art_normalized)
                    if ch_st_match:
                        filter_ch_num = ch_st_match.group(1)
                        filter_st_num = ch_st_match.group(2)
                    else:
                        # Ищем просто "ст. Z"
                        st_match = re.search(r'ст\s*(\d+)', art_normalized)
                        if st_match:
                            filter_st_num = st_match.group(1)
                
                for prov in legal_provisions:
                    prov_normalized = prov.lower().replace('.', ' ').replace('  ', ' ').strip()
                    
                    # Извлекаем компоненты из провизии
                    prov_st_num = None
                    prov_ch_num = None
                    prov_p_num = None
                    
                    # Ищем "п. X ч. Y ст. Z"
                    p_ch_st_match_prov = re.search(r'п\s*(\d+)\s*ч\s*(\d+)\s*ст\s*(\d+)', prov_normalized)
                    if p_ch_st_match_prov:
                        prov_p_num = p_ch_st_match_prov.group(1)
                        prov_ch_num = p_ch_st_match_prov.group(2)
                        prov_st_num = p_ch_st_match_prov.group(3)
                    else:
                        # Ищем "ч. Y ст. Z"
                        ch_st_match_prov = re.search(r'ч\s*(\d+)\s*ст\s*(\d+)', prov_normalized)
                        if ch_st_match_prov:
                            prov_ch_num = ch_st_match_prov.group(1)
                            prov_st_num = ch_st_match_prov.group(2)
                        else:
                            # Ищем просто "ст. Z"
                            st_match_prov = re.search(r'ст\s*(\d+)', prov_normalized)
                            if st_match_prov:
                                prov_st_num = st_match_prov.group(1)
                    
                    # Логика совпадения:
                    # 1. Если фильтр "ст. 5" - совпадает с любой провизией где ст = 5
                    # 2. Если фильтр "ч. 3 ст. 5" - совпадает с провизией где ст = 5 И ч = 3 (независимо от п)
                    # 3. Если фильтр "п. 1 ч. 3 ст. 5" - совпадает только если все компоненты совпадают
                    
                    match = False
                    
                    if filter_st_num and prov_st_num:
                        # Статья совпадает
                        if filter_st_num == prov_st_num:
                            # Если фильтр содержит часть статьи
                            if filter_ch_num:
                                if prov_ch_num and filter_ch_num == prov_ch_num:
                                    # Часть совпадает
                                    if filter_p_num:
                                        # Фильтр содержит пункт - нужно точное совпадение
                                        if prov_p_num and filter_p_num == prov_p_num:
                                            match = True
                                    else:
                                        # Фильтр без пункта - совпадает с любой провизией с этой частью
                                        match = True
                            else:
                                # Фильтр только "ст. X" - совпадает с любой провизией этой статьи
                                match = True
                    
                    if match:
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                continue
        
        filtered.append((idx, score))
    
    return filtered


def keyword_search(query: str, top_k: int = 200) -> List[tuple]:
    """Поиск по ключевым словам в текстовых полях."""
    if not cases:
        return []
    
    # Ищем в полях для переранжирования
    search_fields = ['violation_summary', 'ad_description', 'FAS_arguments', 'ad_content_cited', 'legal_provisions']
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    scores = []
    for idx, case in enumerate(cases):
        score = 0
        for field in search_fields:
            text = case.get(field, '') or ''
            text_lower = text.lower()
            
            if query_lower in text_lower:
                score += 10
            
            for word in query_words:
                if len(word) > 3 and word in text_lower:
                    score += 1
        
        if score > 0:
            scores.append((idx, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def semantic_search(query_embedding: np.ndarray, top_k: int) -> List[tuple]:
    """
    Семантический поиск по косинусному сходству.
    Использует embeddings_FAS_arguments для первичного отбора.
    """
    if embeddings_fas_args is None:
        return []
    
    # Нормализация с защитой от деления на ноль
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        return []  # Возвращаем пустой результат для нулевого вектора
    
    query_norm = query_embedding / norm
    
    # Поиск по FAS_arguments
    similarities = np.dot(embeddings_fas_args, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def rerank_with_field_embeddings(candidates: List[tuple], query_embedding: np.ndarray, use_keyword_scores: bool = False) -> List[dict]:
    """
    Переранжирование кандидатов с использованием эмбеддингов полей.
    FAS_arguments уже НЕ используется - он был для первичного поиска.
    """
    if not cases:
        return []
    
    # Нормализация с защитой от деления на ноль
    norm = np.linalg.norm(query_embedding)
    is_zero_embedding = (norm == 0)
    
    # Если эмбеддинг нулевой и нет семантических результатов - используем только keyword scores
    if is_zero_embedding and use_keyword_scores:
        if not candidates:
            return []
        
        max_keyword_score = max(score for _, score in candidates) if candidates else 1
        if max_keyword_score == 0:
            max_keyword_score = 1
        
        results = []
        for idx, keyword_score in candidates:
            case = cases[idx]
            normalized_score = keyword_score / max_keyword_score
            
            results.append({
                'index': idx,
                'score': normalized_score,
                'keyword_score': keyword_score,
                'field_scores': {
                    'keyword': normalized_score
                },
                'case': case
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    # Стандартная логика с эмбеддингами
    if is_zero_embedding:
        results = []
        for idx, base_score in candidates:
            case = cases[idx]
            results.append({
                'index': idx,
                'score': 0.0,
                'base_score': base_score,
                'field_scores': {},
                'case': case
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    query_norm = query_embedding / norm
    
    results = []
    max_weight_sum = sum(FIELD_WEIGHTS.values())
    
    for idx, base_score in candidates:
        case = cases[idx]
        field_scores = {}
        
        # violation_summary - для переранжирования
        if embeddings_violation is not None and idx < len(embeddings_violation):
            viol_emb = embeddings_violation[idx]
            viol_norm = viol_emb / np.linalg.norm(viol_emb)
            if np.any(viol_emb):
                r = np.dot(query_norm, viol_norm)
                field_scores['violation_summary'] = normalize_score(r)
            else:
                field_scores['violation_summary'] = 0.0
        else:
            field_scores['violation_summary'] = 0.0
        
        # ad_description - для переранжирования
        if embeddings_ad_desc is not None and idx < len(embeddings_ad_desc):
            ad_emb = embeddings_ad_desc[idx]
            ad_norm = ad_emb / np.linalg.norm(ad_emb)
            if np.any(ad_emb):
                r = np.dot(query_norm, ad_norm)
                field_scores['ad_description'] = normalize_score(r)
            else:
                field_scores['ad_description'] = 0.0
        else:
            field_scores['ad_description'] = 0.0
        
        # Взвешенная сумма (FAS_arguments уже не участвует!)
        weighted_sum = sum(
            FIELD_WEIGHTS.get(field, 0) * score 
            for field, score in field_scores.items()
        )
        
        final_score = normalize_score(weighted_sum, min_val=0.0, max_val=max_weight_sum)
        
        results.append({
            'index': idx,
            'score': final_score,
            'base_score': base_score,
            'field_scores': field_scores,
            'case': case
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Гибридный поиск по решениям ФАС."""
    global use_gemini, embeddings_fas_args, cases
    
    if embeddings_fas_args is None or cases is None:
        raise HTTPException(
            status_code=503, 
            detail="Сервер не готов. Данные не загружены."
        )
    
    filters = {}
    if request.year:
        filters['year'] = request.year
    if request.region:
        filters['region'] = request.region
    if request.industry:
        filters['industry'] = request.industry
    if request.article:
        filters['article'] = request.article
    
    # Создаем эмбеддинг запроса
    if use_gemini:
        print(f"Создание эмбеддинга для запроса: {request.query[:50]}...")
        query_embedding = get_embedding(request.query, task_type="retrieval_query")
        
        if query_embedding is None:
            # Gemini недоступен - используем zero-vector
            print("⚠️ Gemini недоступен, используем нулевой эмбеддинг")
            query_embedding = np.zeros(EMBEDDING_DIMENSION)
            use_gemini = False
    else:
        # Используем нулевой эмбеддинг - будет работать только keyword search
        query_embedding = np.zeros(EMBEDDING_DIMENSION)
    
    # Семантический поиск по FAS_arguments (первичный отбор)
    semantic_results = semantic_search(query_embedding, SEARCH_TOP_CANDIDATES)
    
    # Keyword search - работает всегда
    keyword_results = keyword_search(request.query, top_k=50)
    
    # Объединение результатов
    combined_scores = {}
    for idx, score in semantic_results:
        combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
    
    for idx, score in keyword_results:
        combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.3
    
    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:SEARCH_TOP_CANDIDATES]
    
    # Применение фильтров
    if filters:
        filtered_candidates = apply_filters(sorted_candidates, filters)
    else:
        filtered_candidates = sorted_candidates
    
    # Определяем, использовать ли keyword scores для оценок
    use_keyword = len(semantic_results) == 0 and len(keyword_results) > 0
    
    # Переранжирование - передаем флаг use_keyword_scores
    reranked = rerank_with_field_embeddings(filtered_candidates, query_embedding, use_keyword_scores=use_keyword)
    final_results = reranked[:request.top_k]
    
    # Формирование ответа
    case_results = []
    for result in final_results:
        case_data = result['case'].copy()
        case_data['score'] = round(result['score'], 4)
        case_data['field_scores'] = {k: round(v, 4) for k, v in result.get('field_scores', {}).items()}
        case_results.append(CaseResult(**case_data))
    
    return SearchResponse(
        query=request.query,
        total_cases=len(cases),
        results=case_results,
        filters_applied=filters if filters else None,
        message=None
    )


def normalize_industry_name(name: str) -> str:
    """Нормализовать название отрасли - убрать лишние пробелы и т.д."""
    return ' / '.join([s.strip() for s in name.split('/')])


def build_industry_hierarchy(industries: set) -> List[IndustryGroup]:
    """
    Построить иерархию отраслей (до 3 уровней).
    Формат: "Отрасль" или "Отрасль / Подотрасль" или "Отрасль / Подотрасль / Подподотрасль"
    """
    # Сначала подсчитаем количество для каждой отрасли
    industry_counts: Dict[str, int] = {}
    for industry in industries:
        if not industry:
            continue
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    # Теперь строим иерархию
    # all_industries: отрасль -> подотрасль -> {подподотрасль: count}
    all_industries: Dict[str, Dict[str, Dict[str, int]]] = {}
    
    for industry in industries:
        if not industry:
            continue
        
        parts = [s.strip() for s in industry.split('/')]
        
        if len(parts) == 1:
            main = parts[0]
            if main not in all_industries:
                all_industries[main] = {}
        elif len(parts) == 2:
            main, sub = parts
            if main not in all_industries:
                all_industries[main] = {}
            if sub not in all_industries[main]:
                all_industries[main][sub] = {}
        elif len(parts) >= 3:
            main, sub, subsub = parts[0], parts[1], parts[2]
            if main not in all_industries:
                all_industries[main] = {}
            if sub not in all_industries[main]:
                all_industries[main][sub] = {}
            if subsub:
                all_industries[main][sub][subsub] = all_industries[main][sub].get(subsub, 0) + 1
    
    # Строим результат с правильным подсчетом
    result: List[IndustryGroup] = []
    
    for main in sorted(all_industries.keys()):
        subs = all_industries[main]
        sub_list: List[SubIndustry] = []
        total = 0
        
        for sub_name in sorted(subs.keys()):
            sub_data = subs[sub_name]
            
            # Подсчитываем общее количество для подотрасли
            # Суммируем все подподотрасли + саму подотрасль (если есть записи "отрасль / подотрасль" без подподотрасли)
            sub_count = sum(sub_data.values())
            
            # Также добавляем записи вида "Отрасль / Подотрасль" (без третьего уровня)
            direct_key = f"{main} / {sub_name}"
            sub_count += industry_counts.get(direct_key, 0)
            
            # Создаем подподотрасли
            if sub_data:
                sub_sub_list = [SubIndustry(name=ss, count=cc) for ss, cc in sorted(sub_data.items())]
                sub_list.append(SubIndustry(
                    name=sub_name,
                    count=sub_count,
                    sub_industries=sub_sub_list
                ))
            else:
                sub_list.append(SubIndustry(name=sub_name, count=sub_count, sub_industries=[]))
            
            total += sub_count
        
        # Добавляем также записи вида "Отрасль" (без подотрасли)
        total += industry_counts.get(main, 0)
        
        result.append(IndustryGroup(
            name=main,
            count=total,
            sub_industries=sub_list
        ))
    
    return result


# Маппинг названий УФАС к субъектам РФ
UFAS_TO_REGION = {
    "Адыгейское УФАС": "Республика Адыгея",
    "Алтайское УФАС": "Республика Алтай",
    "Амурское УФАС": "Амурская область",
    "Архангельское УФАС": "Архангельская область",
    "Астраханское УФАС": "Астраханская область",
    "Башкортостанское УФАС": "Республика Башкортостан",
    "Белгородское УФАС": "Белгородская область",
    "Брянское УФАС": "Брянская область",
    "Бурятское УФАС": "Республика Бурятия",
    "Волгоградское УФАС": "Волгоградская область",
    "Вологодское УФАС": "Вологодская область",
    "Воронежское УФАС": "Воронежская область",
    "Дагестанское УФАС": "Республика Дагестан",
    "Донецкое УФАС": "Донецкая Народная Республика",
    "Еврейское УФАС": "Еврейская автономная область",
    "Забайкальское УФАС": "Забайкальский край",
    "Ивановское УФАС": "Ивановская область",
    "Ингушское УФАС": "Республика Ингушетия",
    "Иркутское УФАС": "Иркутская область",
    "Кабардино-Балкарское УФАС": "Кабардино-Балкарская Республика",
    "Калининградское УФАС": "Калининградская область",
    "Калмыцкое УФАС": "Республика Калмыкия",
    "Камчатское УФАС": "Камчатский край",
    "Карачаево-Черкесское УФАС": "Карачаево-Черкесская Республика",
    "Карельское УФАС": "Республика Карелия",
    "Кемеровское УФАС": "Кемеровская область",
    "Кировское УФАС": "Кировская область",
    "Коми УФАС": "Республика Коми",
    "Костромское УФАС": "Костромская область",
    "Краснодарское УФАС": "Краснодарский край",
    "Красноярское УФАС": "Красноярский край",
    "Крымское МУФАС": "Республика Крым",
    "Курганское УФАС": "Курганская область",
    "Курское УФАС": "Курская область",
    "Ленинградское УФАС": "Ленинградская область",
    "Липецкое УФАС": "Липецкая область",
    "Луганское УФАС": "Луганская Народная Республика",
    "Магаданское УФАС": "Магаданская область",
    "Марийское УФАС": "Республика Марий Эл",
    "Мордовское УФАС": "Республика Мордовия",
    "Московское УФАС": "Город федерального значения Москва",
    "Московское областное УФАС": "Московская область",
    "Мурманское УФАС": "Мурманская область",
    "Ненецкое УФАС": "Ненецкий автономный округ",
    "Нижегородское УФАС": "Нижегородская область",
    "Новгородское УФАС": "Новгородская область",
    "Новосибирское УФАС": "Новосибирская область",
    "Омское УФАС": "Омская область",
    "Оренбургское УФАС": "Оренбургская область",
    "Орловское УФАС": "Орловская область",
    "Пензенское УФАС": "Пензенская область",
    "Пермское УФАС": "Пермский край",
    "Приморское УФАС": "Приморский край",
    "Псковское УФАС": "Псковская область",
    "Ростовское УФАС": "Ростовская область",
    "Рязанское УФАС": "Рязанская область",
    "Самарское УФАС": "Самарская область",
    "Саратовское УФАС": "Саратовская область",
    "Сахалинское УФАС": "Сахалинская область",
    "Свердловское УФАС": "Свердловская область",
    "Северо-Осетинское УФАС": "Республика Северная Осетия — Алания",
    "Смоленское УФАС": "Смоленская область",
    "Ставропольское УФАС": "Ставропольский край",
    "Тамбовское УФАС": "Тамбовская область",
    "Татарстанское УФАС": "Республика Татарстан",
    "Тверское УФАС": "Тверская область",
    "Томское УФАС": "Томская область",
    "Тульское УФАС": "Тульская область",
    "Тюменское УФАС": "Тюменская область",
    "Удмуртское УФАС": "Удмуртская Республика",
    "Ульяновское УФАС": "Ульяновская область",
    "Хабаровское УФАС": "Хабаровский край",
    "Хакасское УФАС": "Республика Хакасия",
    "Ханты-Мансийское УФАС": "Ханты-Мансийский автономный округ — Югра",
    "Челябинское УФАС": "Челябинская область",
    "Чеченское УФАС": "Чеченская Республика",
    "Чувашское УФАС": "Чувашская Республика",
    "Ямало-Ненецкое УФАС": "Ямало-Ненецкий автономный округ",
    "Ярославское УФАС": "Ярославская область",
    "Севастопольское УФАС": "Город федерального значения Севастополь",
    "Санкт-Петербургское УФАС": "Город федерального значения Санкт-Петербург",
    "Якутское УФАС": "Республика Саха (Якутия)",
}


def build_article_hierarchy(articles: set) -> List[ArticleGroup]:
    """
    Построить иерархию статей закона.
    Формат: "ст. X" -> "ч. 1 ст. X", "ч. 2 ст. X" и т.д.
    """
    # Подсчитываем количество для каждой статьи
    article_counts: Dict[str, int] = {}
    for article in articles:
        if not article:
            continue
        article_counts[article] = article_counts.get(article, 0) + 1
    
    # Группируем по статьям
    # article_map: { "ст. 12": { "ч. 1": 5, "ч. 2": 3 } }
    article_map: Dict[str, Dict[str, int]] = {}
    
    for article in articles:
        if not article:
            continue
        
        # Проверяем формат: "ст. XX" или "ч. N ст. XX"
        main_match = re.match(r'^ст\.\s*(\d+)$', article, re.IGNORECASE)
        part_match = re.match(r'^ч\.\s*(\d+)\s*ст\.\s*(\d+)$', article, re.IGNORECASE)
        
        if main_match:
            # Это основная статья "ст. 12"
            article_num = main_match.group(1)
            main_key = f"ст. {article_num}"
            if main_key not in article_map:
                article_map[main_key] = {}
        elif part_match:
            # Это часть статьи "ч. 1 ст. 12"
            part_num = part_match.group(1)
            article_num = part_match.group(2)
            main_key = f"ст. {article_num}"
            part_key = f"ч. {part_num}"
            
            if main_key not in article_map:
                article_map[main_key] = {}
            article_map[main_key][part_key] = article_counts.get(article, 0)
    
    # Сортируем статьи по номеру
    def article_sort_key(s: str):
        match = re.match(r'^ст\.\s*(\d+)$', s, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    
    sorted_articles = sorted(article_map.keys(), key=article_sort_key)
    
    result: List[ArticleGroup] = []
    for main_key in sorted_articles:
        parts_map = article_map[main_key]
        
        # Подсчитываем общее количество для статьи
        total = article_counts.get(main_key, 0)
        total += sum(parts_map.values())
        
        # Сортируем части по номеру
        def part_sort_key(s: str):
            match = re.match(r'^ч\.\s*(\d+)$', s, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        
        sorted_parts = sorted(parts_map.keys(), key=part_sort_key)
        parts_list = [ArticlePart(name=p, count=parts_map[p]) for p in sorted_parts]
        
        result.append(ArticleGroup(
            name=main_key,
            count=total,
            parts=parts_list
        ))
    
    return result


def build_region_hierarchy(db_regions: set) -> List[RegionGroup]:
    """
    Построить иерархию регионов по федеральным округам.
    Сопоставляет регионы из БД с иерархией ФО.
    """
    result: List[RegionGroup] = []
    
    # Подсчитываем количество дел для каждого региона (УФАС)
    region_counts: Dict[str, int] = {}
    for r in db_regions:
        if r:
            region_counts[r] = region_counts.get(r, 0) + 1
    
    # Создаем обратный маппинг: субъект РФ -> УФАС
    region_to_ufas = {v: k for k, v in UFAS_TO_REGION.items()}
    
    # Проходим по каждому ФО
    for fo_name, fo_regions in REGION_HIERARCHY.items():
        fo_total = 0
        valid_regions = []
        
        for region in fo_regions:
            # Ищем регион в БД по маппингу
            found_count = 0
            matched_ufas = []
            
            # Сначала ищем по маппингу
            ufas_name = region_to_ufas.get(region)
            if ufas_name and ufas_name in region_counts:
                found_count = region_counts[ufas_name]
                matched_ufas.append(ufas_name)
            
            # Также ищем по частичному совпадению (для регионов не в маппинге)
            for db_region in db_regions:
                if db_region and db_region not in matched_ufas:
                    if region.lower() in db_region.lower() or db_region.lower() in region.lower():
                        found_count += region_counts.get(db_region, 0)
                        matched_ufas.append(db_region)
            
            if found_count > 0:
                valid_regions.append(f"{region} ({found_count})")
                fo_total += found_count
        
        if valid_regions:
            # Убираем "(N)" из названий регионов - показываем только субъекты РФ
            clean_regions = [r.split(' (')[0] for r in valid_regions]
            result.append(RegionGroup(
                name=fo_name,
                count=fo_total,
                regions=clean_regions
            ))
    
    return result


def build_industry_hierarchy_from_mapping(industry_counts: Dict[str, int]) -> List[IndustryGroup]:
    """
    Построить иерархию отраслей на основе маппинга INDUSTRY_HIERARCHY.
    Подсчитывает количество дел для каждой категории.
    """
    result: List[IndustryGroup] = []
    
    for industry_name, spheres in INDUSTRY_HIERARCHY.items():
        sub_list: List[SubIndustry] = []
        industry_total = 0
        
        for sphere_name, specs in spheres.items():
            sphere_total = 0
            sub_sub_list: List[SubIndustry] = []
            
            for spec in specs:
                # spec - это реальное значение из БД
                count = industry_counts.get(spec, 0)
                if count > 0:
                    sphere_total += count
                    sub_sub_list.append(SubIndustry(name=spec, count=count))
            
            if sphere_total > 0:
                industry_total += sphere_total
                sub_list.append(SubIndustry(
                    name=sphere_name,
                    count=sphere_total,
                    sub_industries=sub_sub_list
                ))
        
        if industry_total > 0:
            result.append(IndustryGroup(
                name=industry_name,
                count=industry_total,
                sub_industries=sub_list
            ))
    
    return result


@app.get("/api/filters", response_model=FilterOptions)
async def get_filter_options():
    """Получить доступные значения для фильтров."""
    if not cases:
        raise HTTPException(status_code=503, detail="Сервер не готов.")
    
    years = set()
    regions = set()
    industries = set()
    articles = set()
    
    # Подсчитываем количество дел для каждого значения отрасли
    industry_counts: Dict[str, int] = {}
    
    for case in cases:
        if case.get('document_date'):
            try:
                year = int(case['document_date'][:4])
                years.add(year)
            except:
                pass
        
        if case.get('FAS_division'):
            regions.add(case['FAS_division'])
        
        if case.get('defendant_industry'):
            industries.add(case['defendant_industry'])
            industry_counts[case['defendant_industry']] = industry_counts.get(case['defendant_industry'], 0) + 1
        
        if case.get('legal_provisions'):
            legal = case['legal_provisions']
            found_articles = re.findall(r'ст\.\s*\d+|ч\.\s*\d+\s*ст\.\s*\d+', legal, re.IGNORECASE)
            for art in found_articles:
                articles.add(art.strip())
    
    # Строим иерархию отраслей на основе маппинга
    industry_groups = build_industry_hierarchy_from_mapping(industry_counts)
    
    # Строим иерархию регионов
    region_groups = build_region_hierarchy(regions)
    
    # Строим иерархию статей
    article_groups = build_article_hierarchy(articles)
    
    return FilterOptions(
        years=sorted(list(years), reverse=True),
        regions=sorted(list(regions)),
        region_groups=region_groups,
        industries=sorted(list(industries)),
        industry_groups=industry_groups,
        articles=sorted(list(articles)),
        article_groups=article_groups
    )


@app.get("/api/health")
async def health_check():
    """Проверка состояния сервера."""
    return {
        "status": "ok",
        "model_loaded": use_gemini,
        "data_loaded": embeddings_fas_args is not None and cases is not None,
        "total_cases": len(cases) if cases else 0,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_model": "gemini-embedding-001" if use_gemini else "local-embeddings"
    }


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "name": "FAS Hybrid Search API",
        "version": "4.0.0",
        "embedding_model": "gemini-embedding-001" if use_gemini else f"local ({EMBEDDING_DIMENSION}d)",
        "docs": "/docs",
        "health": "/api/health",
        "search": "POST /api/search",
        "filters": "GET /api/filters"
    }
