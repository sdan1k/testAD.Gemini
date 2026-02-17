"""
Скрипт подготовки данных: генерация эмбеддингов из CSV файла решений ФАС.
Использует Google Gemini Embedding API (новый SDK google-genai).
Запуск: python prepare_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Новый SDK google-genai
from google import genai
from google.genai import types

from config import Config


# Глобальная переменная для клиента
client = None


def init_genai_client():
    """Инициализация клиента Gemini."""
    global client
    api_key = Config.get_api_key()
    
    client = genai.Client(api_key=api_key)
    
    print(f"Gemini API настроен с ключом: {api_key[:10]}...")
    print(f"Модель: gemini-embedding-001")
    print(f"Размерность эмбеддингов: {Config.EMBEDDING_DIMENSION}")


def get_embedding(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    """
    Создать эмбеддинг для текста через Gemini API.
    
    Args:
        text: Текст для эмбеддинга
        task_type: Тип задачи (retrieval_query или retrieval_document)
    
    Returns:
        numpy array размерностью Config.EMBEDDING_DIMENSION
    """
    if not text or not text.strip():
        return np.zeros(Config.EMBEDDING_DIMENSION)
    
    task_type_map = {
        "retrieval_query": types.EmbedContentTaskType.RETRIEVAL_QUERY,
        "retrieval_document": types.EmbedContentTaskType.RETRIEVAL_DOCUMENT,
    }
    task = task_type_map.get(task_type, types.EmbedContentTaskType.RETRIEVAL_DOCUMENT)
    
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=Config.EMBEDDING_DIMENSION
            )
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        error_msg = str(e)
        # Проверяем на ошибки геолокации
        if "location" in error_msg.lower() or "not supported" in error_msg.lower():
            print(f"\n⚠️ ОШИБКА: Google Gemini API недоступен в вашем регионе!")
            print(f"   Для использования API требуется VPN или прокси.")
            print(f"   Детали: {error_msg[:100]}...")
        elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            print(f"\n⚠️ ОШИБКА: Превышена квота API!")
            print(f"   Детали: {error_msg[:100]}...")
        else:
            print(f"Ошибка при создании эмбеддинга: {error_msg[:100]}...")
        return None  # Возвращаем None чтобы указать на ошибку


def get_embeddings_batch(texts: list[str], task_type: str = "retrieval_document", 
                         retry_on_quota: bool = True) -> np.ndarray:
    """
    Создать эмбеддинги для списка текстов через Gemini API.
    
    Args:
        texts: Список текстов для эмбеддингов
        task_type: Тип задачи
        retry_on_quota: Повторять ли при ошибке квоты
    
    Returns:
        numpy array формы (len(texts), Config.EMBEDDING_DIMENSION)
    """
    if not texts:
        return np.array([])
    
    all_embeddings = []
    failed_count = 0
    batch_size = 100
    
    task_type_map = {
        "retrieval_query": types.EmbedContentTaskType.RETRIEVAL_QUERY,
        "retrieval_document": types.EmbedContentTaskType.RETRIEVAL_DOCUMENT,
    }
    task = task_type_map.get(task_type, types.EmbedContentTaskType.RETRIEVAL_DOCUMENT)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"\nГенерация эмбеддингов для {len(texts)} текстов...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % 5 == 0 or batch_num == 1:
            print(f"  Батч {batch_num}/{total_batches}... (неудачных: {failed_count})")
            sys.stdout.flush()
        
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=Config.EMBEDDING_DIMENSION
                )
            )
            for emb in result.embeddings:
                all_embeddings.append(np.array(emb.values))
                
        except Exception as e:
            error_msg = str(e)
            # При ошибке используем нулевые векторы
            print(f"  Батч {batch_num} ошибка: {error_msg[:50]}...")
            for _ in range(len(batch)):
                all_embeddings.append(np.zeros(Config.EMBEDDING_DIMENSION))
                failed_count += 1
            
            # Если это ошибка квоты - ждем и пробуем еще раз
            if retry_on_quota and failed_count > 10:
                print(f"\n⚠️ Слишком много ошибок API. Ждем 60 секунд...")
                time.sleep(60)
                failed_count = 0
    
    if failed_count > 0:
        print(f"  Внимание: {failed_count} эмбеддингов не удалось создать (использованы нулевые векторы)")
    
    return np.array(all_embeddings)


def load_csv() -> pd.DataFrame:
    """Загрузка CSV файла с решениями ФАС."""
    # Попробовать разные пути к CSV
    possible_paths = [
        Path(__file__).parent.parent / "fas_ad_practice_dataset.csv",
        Path(__file__).parent.parent / "Legal" / "fas_ad_practice_dataset.csv",
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"CSV файл не найден. Проверьте пути: {[str(p) for p in possible_paths]}")
    
    print(f"Загрузка CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    print(f"Загружено {len(df)} записей")
    return df


def prepare_separate_field_texts(df: pd.DataFrame) -> dict:
    """Подготовить отдельные тексты для каждого поля с эмбеддингами."""
    fas_args_texts = []
    violation_texts = []
    ad_desc_texts = []
    
    for _, row in df.iterrows():
        # FAS_arguments - самое важное для первичного поиска
        if pd.notna(row.get("FAS_arguments")):
            args = str(row['FAS_arguments'])
            if "Ключевой тезис:" in args:
                thesis = args.split("Ключевой тезис:")[1].split("Юридическое")[0].strip()
                fas_args_texts.append(thesis)
            else:
                fas_args_texts.append(args[:500])
        else:
            fas_args_texts.append("")
        
        # violation_summary - для переранжирования
        if pd.notna(row.get("violation_summary")):
            violation_texts.append(str(row['violation_summary']))
        else:
            violation_texts.append("")
        
        # ad_description - для переранжирования
        if pd.notna(row.get("ad_description")):
            ad_desc_texts.append(str(row['ad_description']))
        else:
            ad_desc_texts.append("")
    
    return {
        'FAS_arguments': fas_args_texts,
        'violation_summary': violation_texts,
        'ad_description': ad_desc_texts
    }


def generate_embeddings(texts: list[str], task_type: str = "retrieval_document") -> np.ndarray:
    """Генерация эмбеддингов для списка текстов через Google Gemini."""
    embeddings = get_embeddings_batch(texts, task_type)
    
    if embeddings.size > 0:
        print(f"Размерность эмбеддингов: {embeddings.shape}")
    return embeddings


def prepare_cases(df: pd.DataFrame) -> list[dict]:
    """Подготовка данных кейсов для JSON."""
    cases = []
    
    columns = [
        "docId", "Violation_Type", "document_date", "FASbd_link",
        "FAS_division", "violation_found", "defendant_name",
        "defendant_industry", "ad_description", "ad_content_cited",
        "ad_platform", "violation_summary", "FAS_arguments",
        "legal_provisions", "thematic_tags"
    ]
    
    for idx, row in df.iterrows():
        case = {"index": idx}
        for col in columns:
            value = row.get(col)
            if pd.isna(value):
                case[col] = None
            else:
                case[col] = str(value)
        cases.append(case)
    
    return cases


def main():
    """Главная функция подготовки данных."""
    # Инициализация Gemini API
    init_genai_client()
    
    # Создаем директорию для данных
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Загружаем CSV
    df = load_csv()
    
    # Генерируем отдельные эмбеддинги для полей
    # (по новой архитектуре - не генерируем объединённый embeddings.npy)
    print("\n=== Генерация эмбеддингов для полей ===")
    field_texts = prepare_separate_field_texts(df)
    
    for field_name, field_texts_list in field_texts.items():
        print(f"  - {field_name}...")
        field_embeddings = generate_embeddings(field_texts_list, task_type="retrieval_document")
        field_path = data_dir / f"embeddings_{field_name}.npy"
        np.save(field_path, field_embeddings)
        print(f"    Сохранены: {field_path}")
    
    # Сохраняем кейсы
    cases = prepare_cases(df)
    cases_path = data_dir / "cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Кейсы сохранены: {cases_path}")
    
    print("\n" + "=" * 50)
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
    print("=" * 50)
    print(f"  - Эмбеддинги полей: 3 файла")
    print(f"  - Кейсы: {len(cases)} записей")
    print(f"  - Модель: gemini-embedding-001")
    print(f"  - Размерность: {Config.EMBEDDING_DIMENSION}")


if __name__ == "__main__":
    main()
