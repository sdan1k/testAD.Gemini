"""
FastAPI сервер для гибридного поиска по решениям ФАС.
Запуск: uvicorn main:app --reload --port 8000

Использует Google Gemini Embedding API (gemini-embedding-001) для создания эмбеддингов.
При недоступности Gemini - использует локальные эмбеддинги.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Используем google.generativeai
import google.generativeai as genai

from config import Config

# Конфигурация
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_FAS_ARGS_PATH = DATA_DIR / "embeddings_FAS_arguments.npy"
EMBEDDINGS_VIOLATION_PATH = DATA_DIR / "embeddings_violation_summary.npy"
EMBEDDINGS_AD_DESC_PATH = DATA_DIR / "embeddings_ad_description.npy"
CASES_PATH = DATA_DIR / "cases.json"

# Модель Gemini Embedding 001
MODEL_NAME = "gemini-embedding-001"
# Определим размерность автоматически при загрузке
EMBEDDING_DIMENSION: int = 768

# Веса для полей (для RAG)
FIELD_WEIGHTS = {
    'FAS_arguments': 1.0,
    'violation_summary': 0.8,
    'ad_description': 0.6,
    'ad_content_cited': 0.7,
    'legal_provisions': 0.5
}

# Количество кандидатов для переранжирования
SEARCH_TOP_CANDIDATES = 100


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


def get_embedding(text: str, task_type: str = "retrieval_query") -> np.ndarray:
    """
    Создать эмбеддинг для текста через Gemini API.
    При ошибке возвращает None.
    """
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIMENSION)
    
    try:
        result = genai.embed_content(
            model=MODEL_NAME,
            content=text,
            task_type=task_type
        )
        return np.array(result['embedding'])
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
embeddings: Optional[np.ndarray] = None
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


class FilterOptions(BaseModel):
    years: List[int]
    regions: List[str]
    industries: List[str]
    articles: List[str]


# FastAPI приложение
app = FastAPI(
    title="FAS Hybrid Search API",
    description="API для гибридного поиска по решениям ФАС о нарушениях в рекламе",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure_gemini():
    """Настройка Gemini API."""
    global api_configured
    api_key = Config.get_api_key()
    genai.configure(api_key=api_key)
    api_configured = True
    print(f"Gemini API настроен с ключом: {api_key[:10]}...")


def load_data():
    """Загрузка всех данных при старте."""
    global embeddings, embeddings_fas_args, embeddings_violation, embeddings_ad_desc, cases, EMBEDDING_DIMENSION, use_gemini
    
    print("=" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)
    
    # Пробуем настроить Gemini API
    try:
        configure_gemini()
    except Exception as e:
        print(f"Не удалось настроить Gemini: {e}")
        use_gemini = False
    
    # Основные эмбеддинги
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        EMBEDDING_DIMENSION = embeddings.shape[1]
        print(f"  Основные эмбеддинги: {embeddings.shape}")
        print(f"  Размерность: {EMBEDDING_DIMENSION}")
        
        # Проверяем, соответствует ли размерность Gemini (768)
        if EMBEDDING_DIMENSION != 768:
            print(f"  ⚠️ Размерность эмбеддингов ({EMBEDDING_DIMENSION}) не соответствует Gemini (768)")
            print(f"  → Используем эмбеддинги как есть, Gemini будет недоступен")
            use_gemini = False
    else:
        print(f"  ВНИМАНИЕ: Файл {EMBEDDINGS_PATH} не найден!")
        if not use_gemini:
            print(f"  Запустите python prepare_data.py для создания эмбеддингов")
    
    # Отдельные эмбеддинги для полей
    if EMBEDDINGS_FAS_ARGS_PATH.exists():
        embeddings_fas_args = np.load(EMBEDDINGS_FAS_ARGS_PATH)
        print(f"  FAS_arguments эмбеддинги: {embeddings_fas_args.shape}")
    
    if EMBEDDINGS_VIOLATION_PATH.exists():
        embeddings_violation = np.load(EMBEDDINGS_VIOLATION_PATH)
        print(f"  violation_summary эмбеддинги: {embeddings_violation.shape}")
    
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
        print(f"Режим: Локальные эмбеддинги (размерность {EMBEDDING_DIMENSION})")
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
            if not case.get('defendant_industry') or case['defendant_industry'] not in filters['industry']:
                continue
        
        if filters.get('article'):
            if not case.get('legal_provisions'):
                continue
            legal = case['legal_provisions']
            found = False
            for art in filters['article']:
                if art in legal:
                    found = True
                    break
            if not found:
                continue
        
        filtered.append((idx, score))
    
    return filtered


def keyword_search(query: str, top_k: int = 200) -> List[tuple]:
    """Поиск по ключевым словам в текстовых полях."""
    if not cases:
        return []
    
    search_fields = ['FAS_arguments', 'violation_summary', 'ad_description', 'ad_content_cited', 'legal_provisions']
    
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
    """Семантический поиск по косинусному сходству."""
    if embeddings is None:
        return []
    
    # Нормализация с защитой от деления на ноль
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        return []  # Возвращаем пустой результат для нулевого вектора
    
    query_norm = query_embedding / norm
    
    similarities = np.dot(embeddings, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def rerank_with_field_embeddings(candidates: List[tuple], query_embedding: np.ndarray, use_keyword_scores: bool = False) -> List[dict]:
    """Переранжирование кандидатов с использованием отдельных эмбеддингов полей."""
    if not cases:
        return []
    
    # Нормализация с защитой от деления на ноль
    norm = np.linalg.norm(query_embedding)
    is_zero_embedding = (norm == 0)
    
    # Если эмбеддинг нулевой и нет семантических результатов - используем только keyword scores
    if is_zero_embedding and use_keyword_scores:
        # Нормализуем keyword scores в диапазон 0-1
        if not candidates:
            return []
        
        # Находим max для нормализации
        max_keyword_score = max(score for _, score in candidates) if candidates else 1
        if max_keyword_score == 0:
            max_keyword_score = 1
        
        results = []
        for idx, keyword_score in candidates:
            case = cases[idx]
            # Нормализуем keyword score
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
        # Для нулевого вектора используем только базовые оценки
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
        
        if embeddings_fas_args is not None and idx < len(embeddings_fas_args):
            fas_emb = embeddings_fas_args[idx]
            fas_norm = fas_emb / np.linalg.norm(fas_emb)
            if np.any(fas_emb):
                r = np.dot(query_norm, fas_norm)
                field_scores['FAS_arguments'] = normalize_score(r)
            else:
                field_scores['FAS_arguments'] = 0.0
        else:
            field_scores['FAS_arguments'] = normalize_score(base_score)
        
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
    global use_gemini, embeddings, cases
    
    if embeddings is None or cases is None:
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
    
    # Семантический поиск
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


@app.get("/api/filters", response_model=FilterOptions)
async def get_filter_options():
    """Получить доступные значения для фильтров."""
    if not cases:
        raise HTTPException(status_code=503, detail="Сервер не готов.")
    
    years = set()
    regions = set()
    industries = set()
    articles = set()
    
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
        
        if case.get('legal_provisions'):
            legal = case['legal_provisions']
            found_articles = re.findall(r'ст\.\s*\d+|ч\.\s*\d+\s*ст\.\s*\d+', legal, re.IGNORECASE)
            for art in found_articles:
                articles.add(art.strip())
    
    return FilterOptions(
        years=sorted(list(years), reverse=True),
        regions=sorted(list(regions)),
        industries=sorted(list(industries)),
        articles=sorted(list(articles))
    )


@app.get("/api/health")
async def health_check():
    """Проверка состояния сервера."""
    return {
        "status": "ok",
        "model_loaded": use_gemini,
        "data_loaded": embeddings is not None and cases is not None,
        "total_cases": len(cases) if cases else 0,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_model": "gemini-embedding-001" if use_gemini else "local-embeddings"
    }


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "name": "FAS Hybrid Search API",
        "version": "3.0.0",
        "embedding_model": "gemini-embedding-001" if use_gemini else f"local ({EMBEDDING_DIMENSION}d)",
        "docs": "/docs",
        "health": "/api/health",
        "search": "POST /api/search",
        "filters": "GET /api/filters"
    }
