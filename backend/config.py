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
    # Google API - Gemini Embedding (новый SDK google-genai)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Параметры эмбеддингов - Gemini Embedding 001
    # По умолчанию 3072, но можно переопределить через output_dimensionality
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    
    # Параметры поиска
    DEFAULT_TOP_K = 10
    SEARCH_TOP_CANDIDATES = 100  # Увеличено для первичного отбора
    
    # Веса полей для переранжирования (FAS_arguments уже использован для первичного поиска)
    FIELD_WEIGHTS = {
        # Убрали FAS_arguments - он уже использован в первичном поиске
        'violation_summary': 0.6,
        'ad_description': 0.4
    }
    
    # Параметры базы данных
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Debug режим
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Путь к данным (для развертывания на Render)
    # Если DATA_DIR не задан, используется папка data рядом с этим файлом
    # Для Render: DATA_DIR=/var/data
    DATA_DIR = os.getenv("DATA_DIR")
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """
        Получить директорию с данными.
        Использует DATA_DIR из переменных окружения, если задан,
        иначе - директорию data рядом с этим файлом.
        """
        if cls.DATA_DIR:
            return Path(cls.DATA_DIR)
        return Path(__file__).parent / "data"
    
    @classmethod
    def validate(cls):
        """Проверить наличие обязательных переменных."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY отсутствует в .env файле. "
                "Создайте файл .env с GEMINI_API_KEY=your_api_key\n"
                "Получить ключ можно здесь: https://aistudio.google.com/app/apikey"
            )
    
    @classmethod
    def get_api_key(cls):
        """Получить API ключ."""
        if not cls.GEMINI_API_KEY:
            cls.validate()
        return cls.GEMINI_API_KEY


# Валидация при импорте - отключена, проверка будет при первом вызове
# Config.validate()
