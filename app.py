from services.lang_chain import LangChainService
from services.audio import Audio

from src.document_processor import load_and_process_documents
from src.rag_chain import create_vectorstore, get_retriever, load_vectorstore

# Подготовка документов для RAG
# pdf_path = "./src/data_for_rag.pdf"
# processed_docs = load_and_process_documents(pdf_path)
# vectorstore = create_vectorstore(documents=processed_docs)
vectorstore = load_vectorstore()
retriever = get_retriever(vectorstore)

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
            response = lang_chain_service.rag_assistant(question_text=question, retriever=retriever)
            print(f"Ответ: {response}")
            _audio.text_to_speech(response)

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

def message_assistant():

    x = True

    while x:

        # question = input("Привет! Вы можете задать вопрос отправив текстовое сообщение ниже:")
        question = "При завершении задачи какие статусы выставляются?"

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        try:
            response = lang_chain_service.rag_assistant(question_text=question, retriever=retriever)
            print(f"Ответ: {response}")

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

        x = False


if __name__ == '__main__':

    # audio_assistant()
    message_assistant()
