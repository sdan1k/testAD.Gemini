"""
Конфигурация приложения.
Загружает переменные окружения из .env файла.
"""

from dotenv import load_dotenv
import os
from pathlib import Path

# Загрузить переменные из .env файла в той же директории
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Конфигурация
class Config:
    # Google API - Gemini Embedding 001
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD2mcu_tzdLY26xPGssPTv6nU5MIOwnwis")
    
    # Параметры эмбеддингов - Gemini Embedding 001
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIMENSION = 768  # Gemini embedding-001 имеет размерность 768
    
    # Параметры поиска
    DEFAULT_TOP_K = 10
    SEARCH_TOP_CANDIDATES = 50
    
    # Веса полей для взвешенной релевантности
    FIELD_WEIGHTS = {
        'FAS_arguments': 1.0,
        'violation_summary': 0.8,
        'ad_description': 0.6
    }
    
    # Параметры базы данных
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Debug режим
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Проверить наличие обязательных переменных."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY отсутствует в .env файле. "
                "Создайте файл .env с GOOGLE_API_KEY=your_api_key"
            )
    
    @classmethod
    def get_api_key(cls):
        """Получить API ключ."""
        if not cls.GOOGLE_API_KEY:
            cls.validate()
        return cls.GOOGLE_API_KEY


# Валидация при импорте - отключаем для использования встроенного ключа
# Config.validate()
