import os

from src.chroma_vector_builder import process_and_save_data, load_and_process_documents, create_vectorstore

pdf_path = "./src/data_for_rag.pdf"
vectorstore_path = "./vectorstore"


def main():

    # Проверка наличия PDF-файла
    if not os.path.exists(pdf_path):
        print(f"Ошибка: PDF-файл по пути '{pdf_path}' не найден.")
        return

    # Проверка существования директории векторного хранилища
    # Если хранилища нет, то создадим его из имеющегося pdf
    if not os.path.exists(vectorstore_path):
        print(f"Векторное хранилище '{vectorstore_path}' не найдено. Создаём...")

        try:
            processed_docs = load_and_process_documents(pdf_path)
            create_vectorstore(documents=processed_docs)
            print(f"Векторное хранилище успешно создано в '{vectorstore_path}'!")
        except Exception as e:
            print(f"Ошибка при создании векторного хранилища: {e}")
            return

    # Обработка и обновление данных
    try:
        process_and_save_data(pdf_path, vectorstore_path)
        print(f"Данные успешно обновлены и сохранены в '{vectorstore_path}'!")
    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")


if __name__ == "__main__":
    main()
