from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

vectorstore_path = './vectorstore'

def create_vectorstore(splits):
    """Создание векторного хранилища из обработанных документов"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=vectorstore_path)
    return vectorstore

def load_vectorstore(vectorstore_path: str='./vectorstore'):
    """Загрузка векторной базы"""
    return Chroma(persist_directory=vectorstore_path)

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
