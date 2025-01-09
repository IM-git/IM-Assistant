from src.chroma_vector_builder import create_empty_vectorstore

from resources import Prompts


if __name__ == '__main__':
    create_empty_vectorstore(name='history_store', path=Prompts.HISTORY_VECTORSTORE_PATH)
