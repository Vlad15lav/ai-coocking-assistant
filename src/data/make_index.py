import click

from tqdm import tqdm

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


CONTENT_COLUMNS = [
    "Название",
    "Ингредиенты",
    "Пищевая ценность",
    "Тип кухни",
    "Время приготовления",
    "Класс"
]

METADATA_COLUMNS = [
    "Название",
    "Рецепт",
    "Ингредиенты",
    "Пищевая ценность",
    "Тип кухни",
    "Время приготовления",
    "Ссылка",
    "Класс"
]


@click.command()
@click.argument('csv_path', default="data/raw/food-dataset-ru.csv")
@click.argument('path_index', default="data/processed/food_faiss_index")
@click.argument('hf_model', default="sergeyzh/LaBSE-ru-turbo")
def main(csv_path, path_index, hf_model):
    loader = CSVLoader(
        file_path=csv_path,
        encoding="utf-8",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        },
        content_columns=CONTENT_COLUMNS,
        metadata_columns=METADATA_COLUMNS
    )

    data_documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model
    )

    # Создаём векторное хранилище
    db = None
    with tqdm(total=len(data_documents), desc="Ingesting documents") as pbar:
        for d in data_documents:
            if db:
                db.add_documents([d])
            else:
                db = FAISS.from_documents([d], embeddings)
            pbar.update(1)

    # "food_faiss_index"
    db.save_local(path_index)


if __name__ == "__main__":
    main()
