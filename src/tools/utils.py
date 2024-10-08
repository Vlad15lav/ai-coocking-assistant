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
