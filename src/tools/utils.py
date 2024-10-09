import os
import re
import requests
import aiohttp
import logging

from PIL import Image
from duckduckgo_search import DDGS


def clean_input(input_string: str) -> str:
    """Удаляет лишние пробелы и символы для экономия токенов

    Args:
        input_string (str): входной текст

    Returns:
        str: отфильтрованный текст
    """
    clean_string = re.sub(r'(^|[^\w])[\?!]+', r'\1', input_string)

    return clean_string


def format_docs(docs) -> str:
    """Форматирование документов в единную строку

    Args:
        docs: Документы от ретривера

    Returns:
        (str): Строка со всеми документами
    """
    return "\n\n".join([d.page_content for d in docs])


def format_docs_with_links(docs) -> str:
    """Форматирование документов в единную строку и добвляет метаданные

    Args:
        docs: Документы от ретривера

    Returns:
        (str): Строка со всеми документами
    """
    def helper(d):
        return f"{d.page_content}\nПодробнее:{d.metadata['Ссылка']}"

    return "\n\n".join([helper(d) for d in docs])


def search_image(dict_input: dict) -> dict:
    """Поиск изображения по запросу

    Args:
        dict_input (dict): Входная цепочка

    Returns:
        dict: Цепочка с изображением и мета данными
    """
    img, image_link, url = None, None, None
    try:
        urls_images = DDGS().images(
            keywords=dict_input["query"],
            region='ru-ru',
            safesearch='off',
            type_image="photo",
            max_results=1
            )

        image_link = urls_images[0]['image']
        url = urls_images[0]['url'].rstrip("/")
        img = Image.open(requests.get(image_link, stream=True).raw)
    except Exception as e:
        logging.error(e)

    result_dict = {
        "output": img,
        "image": image_link,
        "url": url,
        "task": "Search Image"
        }

    return result_dict


async def spech2text(audio_file) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/openai/" + \
        "whisper-large-v3-turbo"
    headers = {"Authorization": f'Bearer {os.environ["HF_TOKEN"]}'}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            API_URL,
            headers=headers,
            data=audio_file.read()
        ) as response:
            return await response.json()
