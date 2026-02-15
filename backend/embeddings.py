"""
Сервис эмбеддингов.
Использует Google Gemini Embedding API (gemini-embedding-001).
"""

import numpy as np
from typing import List, Optional
import time

# Используем google.generativeai (старая версия)
import google.generativeai as genai
USE_NEW_LIB = False

from config import Config


class EmbeddingService:
    """
    Сервис для создания эмбеддингов через Google Gemini Embedding API.
    Использует модель gemini-embedding-001 с размерностью 768.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация сервиса эмбеддингов.
        """
        # Используем ключ из параметра или из конфигурации
        self.api_key = api_key or Config.get_api_key()
        
        # Инициализация клиента
        if USE_NEW_LIB:
            self.client = genai.Client(api_key=self.api_key)
        else:
            genai.configure(api_key=self.api_key)
        
        # Модель gemini-embedding-001
        self.model_name = "gemini-embedding-001"
        self.dimension = 768  # Gemini embedding-001 имеет размерность 768
        
        print(f"Сервис эмбеддингов инициализирован.")
        print(f"Модель: {self.model_name}")
        print(f"Размерность эмбеддингов: {self.dimension}")
    
    def embed_text(self, text: str, task_type: str = "retrieval_query") -> np.ndarray:
        """
        Создать эмбеддинг для одного текста.
        
        Args:
            text: Текст для эмбеддинга
            task_type: Тип задачи (retrieval_query или retrieval_document)
        
        Returns:
            numpy array размерностью 768
        """
        if not text or not text.strip():
            return np.zeros(self.dimension)
        
        try:
            if USE_NEW_LIB:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    content=text,
                    config=types.EmbedContentTaskType(task_type)
                )
                return np.array(result.embedding.values)
            else:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type
                )
                return np.array(result['embedding'])
        except Exception as e:
            print(f"Ошибка при создании эмбеддинга: {e}")
            return np.zeros(self.dimension)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Создать эмбеддинги для списка текстов (документов).
        
        Gemini API имеет ограничение на количество текстов за один запрос,
        поэтому делаем батчинг.
        
        Args:
            texts: Список текстов для эмбеддингов
        
        Returns:
            numpy array формы (len(texts), 768)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        batch_size = 100  # Ограничение Gemini API
        
        print(f"Генерация эмбеддингов для {len(texts)} текстов...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"  Обработка батча {batch_num}/{total_batches}...")
            
            try:
                # Gemini API может обрабатывать несколько текстов за раз
                batch_embeddings = []
                for text in batch:
                    embedding = self.embed_text(text, task_type="retrieval_document")
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Ошибка при обработке батча {batch_num}: {e}")
                # В случае ошибки возвращаем нулевые эмбеддинги для батча
                all_embeddings.extend([np.zeros(self.dimension).tolist() for _ in range(len(batch))])
        
        return np.array(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Создать эмбеддинг для поискового запроса.
        
        Args:
            query: Поисковый запрос
        
        Returns:
            numpy array размерностью 768
        """
        return self.embed_text(query, task_type="retrieval_query")
    
    def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Создать эмбеддинги для батча текстов.
        
        Args:
            texts: Список текстов
            task_type: Тип задачи
        
        Returns:
            numpy array формы (len(texts), 768)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = []
                for text in batch:
                    embedding = self.embed_text(text, task_type)
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Ошибка при обработке батча: {e}")
                all_embeddings.extend([np.zeros(self.dimension).tolist() for _ in range(len(batch))])
        
        return np.array(all_embeddings)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Нормализовать эмбеддинги (для косинусного сходства).
    
    Args:
        embeddings: Массив эмбеддингов
    
    Returns:
        Нормализованные эмбеддинги
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Вычислить косинусное сходство между двумя векторами.
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
    
    Returns:
        Косинусное сходство (от -1 до 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Нормализовать оценку в диапазон [0, 1].
    
    Args:
        score: Оценка
        min_val: Минимальное значение
        max_val: Максимальное значение
    
    Returns:
        Нормализованная оценка в [0, 1]
    """
    if score < min_val:
        return 0.0
    if score > max_val:
        return 1.0
    return (score - min_val) / (max_val - min_val)
