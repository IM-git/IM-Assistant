from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

path_for_pdf = './src/data_for_rag.pdf'
path_for_vectorstore = './vectorstore'
collection_name = "im_example_collection"

embeddings = OpenAIEmbeddings()


def process_and_save_data(pdf_path: str = path_for_pdf, vectorstore_path: str = path_for_vectorstore):
    """
    Обработка PDF и сохранение векторной базы.
    Args:
        pdf_path: Путь к PDF-файлу для обработки.
        vectorstore_path: Директория для сохранения векторного хранилища.
    """

    chunks = load_and_process_documents(pdf_path)

    # Сохранение данных в векторной базе
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=vectorstore_path)

    print(f"Векторное хранилище сохранено в {vectorstore_path}")

    return vectorstore

def load_and_process_documents(pdf_path: str):
    """Загружает и обрабатывает PDF-файлы"""

    # Загрузка документа
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Разделение документа на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return splits

def create_vectorstore(documents, path: str=path_for_vectorstore, name: str=collection_name):
    """Создание векторного хранилища из обработанных документов"""
    vectorstore = Chroma.from_documents(documents=documents,
                                        embedding=embeddings,
                                        collection_name=name,
                                        persist_directory=path)
    return vectorstore

def load_vectorstore(path: str=path_for_vectorstore, name: str=collection_name):
    """Загрузка существующей векторной базы"""
    return Chroma(collection_name=name,
                  embedding_function=embeddings,
                  persist_directory=path)

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 5})
