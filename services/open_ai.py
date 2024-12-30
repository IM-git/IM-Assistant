import os
import pyttsx3
import speech_recognition as sr
from openai import OpenAI

# Настройка OpenAI API
OPENAI_API_KEY = ''
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

GPT_MODEL = 'gpt-4o'

# Функция для преобразования текста в голос
def text_to_speech(text: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)  # Настройка скорости речи
    engine.setProperty('volume', 1.0)  # Настройка громкости
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Установить голос
    engine.say(text)
    engine.runAndWait()


# Функция для преобразования голоса в текст
def speech_to_text() -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Скажите что-нибудь...")
        recognizer.adjust_for_ambient_noise(source)  # Удаление фонового шума
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")  # Язык: "en-US" для английского
        return text
    except sr.UnknownValueError:
        print("Речь не распознана. Попробуйте еще раз.")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")
    return ""


# Функция для обработки запроса от пользователя
def user_type_assistant(question_text: str):
    return client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": question_text
            }
        ]
    ).choices[0].message.content


def developer_type_assistant(question_text: str, content: str):
    return client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "developer",
                "content": [{"type": "text", "text": content}]
            },
            {
                "role": "user",
                "content": question_text
            }
        ]
    ).choices[0].message.content


def assistant_type_assistant(question_text, context, content):
    return client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "knock knock."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Who's there?"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Orange."}]
            }
        ]
    ).choices[0].message.content


if __name__ == '__main__':

    print("Привет! Вы можете задать вопрос голосом.")
    _developer_assistant_context = ("You are a helpful assistant that answers programming questions "
                                    "in the style of a southern belle from the southeast United States.")

    while True:
        # Голосовой ввод
        question = speech_to_text()
        if question.lower() in ["выход", "стоп", "закончить"]:
            print("Завершение работы.")
            break

        print(f"Ваш вопрос: {question}")

        # Выбор подходящего метода для ответа
        try:
            response_text = developer_type_assistant(question_text=question, content=_developer_assistant_context)
            print(f"Ответ: {response_text}")

            # Преобразование ответа в голос
            text_to_speech(response_text)
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
