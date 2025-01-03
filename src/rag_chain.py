from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

vectorstore_path = './vectorstore'

def create_vectorstore(splits):
    """Создание векторного хранилища из обработанных документов"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits,
                                        collection_name="im_example_collection",
                                        embedding=embeddings,
                                        persist_directory=vectorstore_path)
    return vectorstore

def load_vectorstore(path: str=vectorstore_path):
    """Загрузка существующей векторной базы"""
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=path,
                  embedding_function=embeddings,
                  collection_name="im_example_collection")

def get_retriever(vectorstore):
    """Настройка извлечения документов"""
    return vectorstore.as_retriever(search_type="similarity",
                                    search_kwargs={"k": 5})
