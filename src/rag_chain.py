from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()
vectorstore_path = './vectorstore'
collection_name = "im_example_collection"

def create_vectorstore(documents, path: str=vectorstore_path, name: str=collection_name):
    """Создание векторного хранилища из обработанных документов"""
    vectorstore = Chroma.from_documents(documents=documents,
                                        embedding=embeddings,
                                        collection_name=name,
                                        persist_directory=path)
    return vectorstore

def load_vectorstore(path: str=vectorstore_path, name: str=collection_name):
    """Загрузка существующей векторной базы"""
    return Chroma(collection_name=name,
                  embedding_function=embeddings,
                  persist_directory=path)

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 5})
