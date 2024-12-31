from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def process_and_save_pdf(pdf_path: str = './src/data_for_rag.pdf', vectorstore_path: str = './vectorstore'):
    """
    Обработка PDF и сохранение векторной базы.

    - pdf_path: Путь к PDF-файлу для обработки.
    - vectorstore_path: Директория для сохранения векторного хранилища.
    """
    # Загрузка документа
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Разбиение на chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Создание векторной базы
    embeddings = OpenAIEmbeddings()

    # Сохранение базы
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=vectorstore_path)

    print(f"Векторное хранилище сохранено в {vectorstore_path}")


def load_vectorstore(vectorstore_path: str):
    """
    Загрузка векторного хранилища из указанной директории.

    - vectorstore_path: Путь к директории векторного хранилища.
    """
    vectorstore = Chroma(persist_directory=vectorstore_path)
    print(f"Векторное хранилище загружено из {vectorstore_path}")
    return vectorstore


def retrieve_relevant_chunks(vectorstore, query: str, k: int = 5, filter_criteria: dict = None):
    """
    Извлечение релевантных chunks на основе запроса пользователя.

    - vectorstore: Загруженное векторное хранилище.
    - query: Запрос пользователя.
    - k: Количество возвращаемых chunks.
    - filter_criteria: Условия фильтрации (метаданные).
    """
    # Настройка извлечения
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": filter_criteria  # Например, {"metadata.section": "summary"}
        }
    )

    # Извлечение данных
    relevant_chunks = retriever.retrieve(query)
    return relevant_chunks

def load_and_process_documents(pdf_path: str):
    """Загружает и обрабатывает PDF-файлы"""
    # Загрузка документа
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Разделение документа на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return splits
