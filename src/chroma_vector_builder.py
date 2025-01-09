from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever

import chromadb

path_for_pdf = './src/data_for_rag.pdf'
path_for_vectorstore = './vectorstore'
collection_name = "im_example_collection"

embeddings = OpenAIEmbeddings()


def process_and_save_data(pdf_path, vectorstore_path, name):
    """
    Обработка PDF и сохранение векторной базы.
    Args:
        pdf_path: Путь к PDF-файлу для обработки.
        vectorstore_path: Директория для сохранения векторного хранилища.
        name: Название для новой/существующей коллекции векторного хранилища.
    """
    try:
        documents = load_and_process_documents(pdf_path)

        # Сохранение данных в векторной базе
        vectorstore = create_vectorstore(documents=documents,
                                         name=name,
                                         path=vectorstore_path)
        print(f"Векторное хранилище сохранено в {vectorstore_path}")
        return vectorstore

    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")

def load_and_process_documents(pdf_path: str):
    """Загружает и обрабатывает PDF-файлы"""

    try:
        # Загрузка документа
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Разделение документа на фрагменты
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return splits

    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")

def create_vectorstore(documents, name: str=collection_name, path: str=path_for_vectorstore):
    """Создание векторного хранилища из обработанных документов"""
    try:
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            collection_name=name,
                                            persist_directory=path)
        return vectorstore
    except Exception as e:
        print(f"Ошибка при создании векторного хранилища: {e}")

def create_empty_vectorstore(name: str=collection_name, path: str=path_for_vectorstore):
    """Создание векторного хранилища из обработанных документов"""
    try:

        vectorstore = Chroma(collection_name=name,
                             persist_directory=path)
        return vectorstore
    except Exception as e:
        print(f"Ошибка при создании векторного хранилища: {e}")

def load_vectorstore(path: str=path_for_vectorstore, name: str=collection_name):
    """Загрузка существующей векторной базы"""
    return Chroma(collection_name=name,
                  embedding_function=embeddings,
                  persist_directory=path)

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 5})

def get_multi_retriever(retrievers: list):
    multi_retriever = MergerRetriever(retrievers=retrievers)
    return multi_retriever
