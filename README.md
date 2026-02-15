**- Backend: FastAPI with Google Gemini embedding-001**
**- Frontend: Next.js 14 with dynamic background effects**
**- Added README with instructions for developer**
**- Added .env.example template**
**- Note: Gemini API key needs to be updated for production use**

________________________________________________________________
read.me:

# Поиск практики ФАС по нарушениям в рекламе

Семантический поиск по базе решений ФАС о нарушениях в рекламе с использованием эмбеддингов Google Gemini.

## ⚠️ ВАЖНО ДЛЯ РАЗРАБОТЧИКА

### Проблемы с Google Gemini API

**Текущее состояние:** API ключ `AIzaSyA_EDbi-T8dHzrNwMDhmKI8NNc2IpyVL20` возвращает ошибку **403 Forbidden**.

**Возможные причины:**
1. API ключ неактивен или истек
2. API не включен в Google Cloud Console
3. Ограничения по региону (нужен VPN для доступа к Google API)
4. Превышен лимит запросов

**Что нужно сделать:**

1. **Получить новый API ключ:**
   - Перейдите на https://aistudio.google.com/app/apikey
   - Создайте новый API ключ
   - Скопируйте его в файл `backend/.env`

2. **Настроить VPN (если нужно):**
   - Google API может требовать VPN для работы из России

3. **Пересоздать эмбеддинги:**
   ```bash
   cd backend
   rm -rf data/embeddings.parquet
   python prepare_data.py
   ```

4. **Проверить работает ли API:**
   ```bash
   python -c "from embeddings import get_embedding; print(get_embedding('тест'))"
   ```

### Структура проекта

```
Legal-main/
├── backend/           # FastAPI сервер
│   ├── config.py    # Конфигурация
│   ├── embeddings.py # Эмбеддинги Gemini
│   ├── main.py      # API endpoints
│   └── prepare_data.py # Подготовка данных
│
├── frontend/         # Next.js приложение
│   └── src/
│       ├── app/    # Страницы
│       └── components/ # UI компоненты
│
└── data/           # Данные ( parquet файлы)
```

## 🚀 Запуск

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Отредактируйте .env с вашим API ключом
python prepare_data.py  # первый раз
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Использование

1. Откройте http://localhost:3000
2. Введите запрос (например: "реклама алкоголя в интернете")
3. Система найдет похожие решения ФАС

## 📝 Технологии

- **Backend:** FastAPI, Google Gemini Embedding 001
- **Frontend:** Next.js 14, React, Tailwind CSS
- **База данных:** SQLite + Parquet для эмбеддингов
- **Поиск:** FAISS для векторного поиска
