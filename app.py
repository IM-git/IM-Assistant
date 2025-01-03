from flask import Flask, request, jsonify

import tiktoken

from services.open_ai_service import OpenAIService
from services.audio import Audio

from src.document_processor import load_and_process_documents
from src.rag_chain import create_vectorstore, get_retriever, load_vectorstore

app = Flask(__name__)
open_ai_service = OpenAIService(gpt_model="gpt-4o-mini")

# Подготовка документов для RAG
pdf_path = "./src/data_for_rag.pdf"
processed_docs = load_and_process_documents(pdf_path)
# vectorstore = create_vectorstore(processed_docs)
vectorstore = load_vectorstore()
retriever = get_retriever(vectorstore)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Вопрос не задан"}), 400

    # Получение ответа через RAG
    response = open_ai_service.rag_assistant(question_text=question, retriever=retriever)
    return jsonify({"response": response})

def audio_assistant():
    _audio = Audio()
    _openai = OpenAIService()

    print("Привет! Вы можете задать вопрос голосом.")
    _developer_assistant_context = ("You are a helpful assistant that answers programming questions "
                                    "in the style of a southern belle from the southeast United States.")

    while True:
        # Голосовой ввод
        question = _audio.speech_to_text()

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        # Выбор подходящего метода для ответа
        try:
            response_text = _openai.developer_type_assistant(question_text=question,
                                                             content=_developer_assistant_context)
            print(f"Ответ: {response_text}")

            # Преобразование ответа в голос
            _audio.text_to_speech(response_text)

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

def message_assistant():

    while True:

        _developer_assistant_context = ("You are a helpful assistant that answers programming questions "
                                        "in the style of a southern belle from the southeast United States.")

        question = input("Привет! Вы можете задать вопрос отправив текстовое сообщение ниже:")

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        try:
            response = open_ai_service.rag_assistant(question_text=question, retriever=retriever)
            print(f"Ответ: {response}")

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """ Returns the number of tokens in a text string.
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



if __name__ == '__main__':

    # audio_assistant()
    message_assistant()
