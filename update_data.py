from src.document_processor import process_and_save_data

if __name__ == "__main__":

    pdf_path = "./src/data_for_rag.pdf"
    vectorstore_path = "./vectorstore"
    process_and_save_data(pdf_path, vectorstore_path)
    print("Данные успешно обновлены!")
