import re


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
