"""
Скрипт подготовки данных: генерация эмбеддингов из CSV файла решений ФАС.
Использует Google Gemini Embedding API (gemini-embedding-001).
Запуск: python prepare_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Используем google.generativeai (старая версия)
import google.generativeai as genai
USE_NEW_LIB = False

from config import Config


# Глобальная переменная для клиента
client = None


def init_genai_client():
    """Инициализация клиента Gemini."""
    global client
    api_key = Config.get_api_key()
    
    if USE_NEW_LIB:
        client = genai.Client(api_key=api_key)
    else:
        genai.configure(api_key=api_key)
    
    print(f"Gemini API настроен с ключом: {api_key[:10]}...")
    print(f"Модель: gemini-embedding-001")
    print(f"Размерность эмбеддингов: 768")


def get_embedding(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    """
    Создать эмбеддинг для текста через Gemini API.
    
    Args:
        text: Текст для эмбеддинга
        task_type: Тип задачи (retrieval_query или retrieval_document)
    
    Returns:
        numpy array размерностью 768
    """
    if not text or not text.strip():
        return np.zeros(768)
    
    try:
        if USE_NEW_LIB:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                content=text,
                config=types.EmbedContentTaskType(task_type)
            )
            return np.array(result.embedding.values)
        else:
            result = genai.embed_content(
                model="gemini-embedding-001",
                content=text,
                task_type=task_type
            )
            return np.array(result['embedding'])
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
        numpy array формы (len(texts), 768)
    """
    if not texts:
        return np.array([])
    
    all_embeddings = []
    failed_count = 0
    batch_size = 100
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"\nГенерация эмбеддингов для {len(texts)} текстов...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if batch_num % 5 == 0 or batch_num == 1:
            print(f"  Батч {batch_num}/{total_batches}... (неудачных: {failed_count})")
            sys.stdout.flush()
        
        batch_embeddings = []
        for text in batch:
            embedding = get_embedding(text, task_type)
            
            if embedding is None:
                # Если получили None (ошибка API), используем нулевой вектор
                embedding = np.zeros(768)
                failed_count += 1
                
                # Если это ошибка квоты - ждем и пробуем еще раз
                if retry_on_quota and failed_count > 10:
                    print(f"\n⚠️ Слишком много ошибок API. Ждем 60 секунд...")
                    time.sleep(60)
                    failed_count = 0
            
            batch_embeddings.append(embedding)
        
        all_embeddings.extend(batch_embeddings)
    
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


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Объединение текстовых полей для создания эмбеддингов."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        
        if pd.notna(row.get("ad_content_cited")):
            parts.append(f"Реклама: {row['ad_content_cited']}")
        
        if pd.notna(row.get("ad_description")):
            parts.append(f"Описание рекламы: {row['ad_description']}")
        
        if pd.notna(row.get("violation_summary")):
            parts.append(f"Нарушение: {row['violation_summary']}")
        
        if pd.notna(row.get("FAS_arguments")):
            args = str(row['FAS_arguments'])
            if "Ключевой тезис:" in args:
                thesis = args.split("Ключевой тезис:")[1].split("Юридическое")[0].strip()
                parts.append(f"Обоснование ФАС: {thesis}")
            else:
                parts.append(f"Обоснование ФАС: {args[:500]}")
        
        if pd.notna(row.get("legal_provisions")):
            legal = str(row['legal_provisions'])
            legal = legal.replace('[', '').replace(']', '').replace("'", '')
            parts.append(f"Нарушенные статьи: {legal}")
        
        if pd.notna(row.get("thematic_tags")):
            tags = str(row['thematic_tags'])
            parts.append(f"Теги: {tags}")
        
        if pd.notna(row.get("defendant_industry")):
            parts.append(f"Отрасль: {row['defendant_industry']}")
        
        if pd.notna(row.get("ad_platform")):
            parts.append(f"Платформа: {row['ad_platform']}")
        
        if pd.notna(row.get("Violation_Type")):
            violation_type = "нарушение содержания" if str(row['Violation_Type']) == "substance" else "нарушение размещения"
            parts.append(f"Тип: {violation_type}")
        
        text = " ".join(parts) if parts else "Нет данных"
        texts.append(text)
    
    return texts


def prepare_separate_field_texts(df: pd.DataFrame) -> dict:
    """Подготовить отдельные тексты для каждого поля с эмбеддингами."""
    fas_args_texts = []
    violation_texts = []
    ad_desc_texts = []
    
    for _, row in df.iterrows():
        if pd.notna(row.get("FAS_arguments")):
            args = str(row['FAS_arguments'])
            if "Ключевой тезис:" in args:
                thesis = args.split("Ключевой тезис:")[1].split("Юридическое")[0].strip()
                fas_args_texts.append(thesis)
            else:
                fas_args_texts.append(args[:500])
        else:
            fas_args_texts.append("")
        
        if pd.notna(row.get("violation_summary")):
            violation_texts.append(str(row['violation_summary']))
        else:
            violation_texts.append("")
        
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
    
    # Подготавливаем тексты
    texts = prepare_texts(df)
    
    # Генерируем основные эмбеддинги
    print("\n=== Генерация основных эмбеддингов ===")
    embeddings = generate_embeddings(texts, task_type="retrieval_document")
    
    if embeddings.size == 0:
        print("❌ Не удалось создать эмбеддинги. Проверьте подключение к VPN.")
        return
    
    # Сохраняем основные эмбеддинги
    embeddings_path = data_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Эмбеддинги сохранены: {embeddings_path}")
    
    # Генерируем отдельные эмбеддинги для полей
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
    print(f"  - Основные эмбеддинги: {embeddings.shape[0]} векторов размерности {embeddings.shape[1]}")
    print(f"  - Кейсы: {len(cases)} записей")
    print(f"  - Модель: gemini-embedding-001")


if __name__ == "__main__":
    main()
