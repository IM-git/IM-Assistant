from src.document_processor import process_and_save_pdf

if __name__ == "__main__":

    pdf_path = "./src/data_for_rag.pdf"
    vectorstore_path = "./vectorstore"
    process_and_save_pdf(pdf_path, vectorstore_path)
    print("Данные успешно обновлены!")
