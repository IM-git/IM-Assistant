from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def process_and_save_pdf(pdf_path: str='./src/data_for_rag.pdf', vectorstore_path: str='./vectorstore'):
    """Обработка PDF и сохранение векторной базы"""

    # Загрузка документа
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Разбиение на chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Создание векторной базы
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # Сохранение базы
    vectorstore.persist(vectorstore_path)

def load_and_process_documents(pdf_path: str):
    """Загружает и обрабатывает PDF-файлы"""
    # Загрузка документа
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Разделение документа на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return splits
