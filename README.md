# AI Coocking Assistant

**AI Cooking Assistant** — это интеллектуальный помощник на базе LLM (Large Language Model), реализованный с использованием фреймворка [LangChain](https://langchain.com).

## Описание проекта

Основное назначение приложения — помочь пользователю с выбором блюда, сгенерировать новое уникальное блюдо или найти фотографию для сравнения.

Сценарии использования:
- Персональная рекомендация блюда.
- AI помощник для официантов в ресторанах по их меню.
- Стандартная задача поиска по запросу пользователя.
- Поиск вдохновения для приготовления блюда.

## Особенности реализации

Основные модели и подходы, которые использовались при реализации:
- Использование модели Llama 3.1 70b от [Groq API](https://console.groq.com) для LLM-агента.
- Создание RAG рецептов с помощью эмбеддингов [LaBSE RU Turbo](https://huggingface.co/sergeyzh/LaBSE-ru-turbo) и векторизации FAISS.
- Написание агента с Router Chain для четырех инструментов с LLM и промптом.
- Поиск изображений с помощью [DuckDuckGo](https://pypi.org/project/duckduckgo-search/#4-images---image-search-by-duckduckgocom) Search API.
- Использование модели [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) от HuggingFace для голосового ввода.
- Реализация приложения для инференса с помощью [Streamlit Cloud](https://streamlit.io/cloud).

Дальнейшие возможные улучшения проекта:
- Добавление памяти для LLM-агента.
- Fine-Tuning LLM на данных рецептов с помощью LoRa.
- Добавление RAG на текстах кулинарных книг.
- Реализация Telegram-бота.

## Настройка окружения
В данном проекте используются модели от сторонних сервисов, поэтому необходимо настроить только окружение:

1. Установите Python 3.11.

2. Клонирование репозитория:
    ```bash
    git clone https://github.com/Vlad15lav/ai-coocking-assistant.git
    cd ai-coocking-assistant
    ```

3. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

4. Настройте API ключи для работы с моделями ([Groq API](https://console.groq.com), [HuggingFace](https://huggingface.co/)).
    ```
    OPENAI_API_KEY=<YOUR_KEY>
    HF_TOKEN=<YOUR_KEY>
    ```

5. Запустите приложение:
    ```bash
    cd ./src
    streamlit run streamlit_app.py
    ```

## Технические особенности

Схема реализации проекта:

<img src="./imgs/Project Scheme.svg">

Пользователь может ввести или отправить голосом свой запрос. Далее данный запрос обрабатывается промпт-классификатором, который определяет, к какой цепочке относится задача:
- Рекомендация рецепта по предпочтениям пользователя.
- Генерация нового рецепта по предпочтениям пользователя.
- Запрос на ознакомление с функциями агента.
- Поиск изображения блюда по его описанию и названию.

Для разработки системы RAG были использованы данные рецептов 37 638 различных блюд, которые были собраны путем парсинга с указанного [сайта](https://www.eda.ru). Программный код парсера из моего проекта по [RecSys](https://github.com/Vlad15lav/food-recsys) доступен [здесь](https://github.com/Vlad15lav/food-recsys/blob/main/notebooks/data-parser.ipynb), а скачать данные можно по [ссылке](https://www.kaggle.com/datasets/vlad15lav/recipes-corpus-textual-data-for-nlprecsys) на платформе Kaggle.

В качестве хранилища эмбеддингов и механизма поиска (Retriever) была выбрана библиотека [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/), что позволяет эффективно обрабатывать и хранить векторы, обеспечивая быстрый доступ к необходимой информации для контекста промпта. Среди нескольких рецептов, предложенных ретривером, LLM рекомендует блюдо или генерирует новое на основе контекста.

Если пользователю необходимо найти фотографию блюда, то сначала LLM фильтрует запрос пользователя для [DuckDuckGo](https://pypi.org/project/duckduckgo-search/#4-images---image-search-by-duckduckgocom) API, которое возвращает изображение из интернета.

## Deploy

Приложение развернуто в [Streamlit Cloud](https://ai-coocking-assistant.streamlit.app/) из-за его простоты и удобства:  
- Развёртывание в один клик.
- Деплой кода прямо из репозитория.
- Мгновенные обновления приложения при каждом изменении кода.
- Простое управление проектом.

## Ссылки на источники
- [Документация LangChain](https://langchain.com/docs)
- [Groq API](https://console.groq.com)
- [HuggingFace](https://huggingface.co/)
- [Creating AI products by ChatGPT](https://stepik.org/course/178846/)
