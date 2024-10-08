import os
import requests
import urllib.parse


def download_from_yandex(public_link: str, save_path: str):
    """Скачивание файла из Yandex диска

    Args:
        public_link (str): Публичная ссылка на файл
        save_path (str): Путь директории куда сохранить файл
    """
    url = "https://cloud-api.yandex.net/v1/disk/public/resources/" + \
        f"download?public_key={public_link}"
    download_url = requests.get(url).json()["href"]
    file_name = urllib.parse.unquote(
        download_url.split("filename=")[1].split("&")[0]
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_save_path = os.path.join(save_path, file_name)

    with open(file_save_path, "wb") as file:
        download_response = requests.get(download_url, stream=True)

        for chunk in download_response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                file.flush()
