from services.lang_chain import LangChainService
from services.audio import Audio

from src.chroma_vector_builder import load_and_process_documents, create_vectorstore, get_retriever, load_vectorstore
from resources import Prompts

PDF_VECTORSTORE_PATH = Prompts.PDF_VECTORSTORE_PATH
HISTORY_VECTORSTORE_PATH = Prompts.HISTORY_VECTORSTORE_PATH

# Подготовка документов для RAG
pdf_path = "./src/data_for_rag.pdf"
processed_docs = load_and_process_documents(pdf_path)

# vectorstore = create_vectorstore(documents=processed_docs)

# Векторная база из pdf
pdf_vectorstore = load_vectorstore(path=Prompts.PDF_VECTORSTORE_PATH)
pdf_retriever = get_retriever(pdf_vectorstore)

# Векторная база из предыдущих вопросов и ответов
# history_vectorstore = load_vectorstore(path=Prompts.HISTORY_VECTORSTORE_PATH)
# history_retriever = get_retriever(history_vectorstore)
#
# retriever = pdf_retriever + history_retriever

lang_chain_service = LangChainService(gpt_model="gpt-4o-mini")


def audio_assistant():

    _audio = Audio()

    print("Привет! Вы можете задать вопрос голосом.")

    while True:

        question = _audio.speech_to_text()

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        try:
            response = lang_chain_service.rag_assistant(question_text=question, retriever=pdf_retriever)
            print(f"Ответ: {response}")
            _audio.text_to_speech(response)

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

def message_assistant():

    x = True

    while x:

        # question = input("Привет! Вы можете задать вопрос отправив текстовое сообщение ниже:")
        # question = "При завершении задачи какие статусы выставляются?"
        question = "Кто такой паладин?"

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        try:
            response = lang_chain_service.rag_assistant(question_text=question, retriever=pdf_retriever)
            print(f"Ответ: {response}")

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

        x = False


if __name__ == '__main__':

    # audio_assistant()
    message_assistant()
