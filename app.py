from services.open_ai_service import OpenAIService
from services.audio import Audio


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
    _openai = OpenAIService()

    while True:

        _developer_assistant_context = ("You are a helpful assistant that answers programming questions "
                                        "in the style of a southern belle from the southeast United States.")

        question = input("Привет! Вы можете задать вопрос отправив текстовое сообщение ниже:")

        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        try:
            response_text = _openai.developer_type_assistant(question_text=question,
                                                             content=_developer_assistant_context)
            print(f"Ответ: {response_text}")

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")



if __name__ == '__main__':

    # audio_assistant()
    message_assistant()
