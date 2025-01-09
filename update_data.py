import os

from src.chroma_vector_builder import process_and_save_data, load_and_process_documents, create_vectorstore

from resources import Prompts

pdf_path = "./src/data_for_rag.pdf"
vectorstore_path = Prompts.PDF_VECTORSTORE_PATH
collection_name = "im_example_collection"


def main():

    # Проверка наличия PDF-файла
    if not os.path.exists(pdf_path):
        print(f"Ошибка: PDF-файл по пути '{pdf_path}' не найден.")
        return

    # Проверка существования директории векторного хранилища
    # Если хранилища нет, то создадим его из имеющегося pdf
    if os.path.exists(vectorstore_path):
        process_and_save_data(pdf_path, vectorstore_path, collection_name)
        print(f"Данные успешно обновлены и сохранены в '{vectorstore_path}'!")

    else:
        print(f"Векторное хранилище '{vectorstore_path}' не найдено. Создаём...")
        process_and_save_data(pdf_path, vectorstore_path, collection_name)
        print(f"Векторное хранилище успешно создано в '{vectorstore_path}'!")


if __name__ == "__main__":
    main()
