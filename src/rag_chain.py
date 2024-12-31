from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_vectorstore(splits):
    """Создание векторного хранилища из обработанных документов"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def load_vectorstore(vectorstore_path: str='./vectorstore'):
    """Загрузка векторной базы"""
    return Chroma(persist_directory=vectorstore_path)

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever()
