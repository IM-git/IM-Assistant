import os
from openai import OpenAI

OPENAI_API_KEY = ''


class OpenAIService:

    def __init__(self, gpt_model: str = 'gpt-4o'):

        self.gpt_model = gpt_model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

    # Функция для обработки запроса от пользователя
    def user_type_assistant(self, question_text: str):
        return self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": question_text
                }
            ]
        ).choices[0].message.content


    def developer_type_assistant(self, question_text: str, content: str):
        return self.client.chat.completions.create(
            model=self.gpt_model,
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


    def assistant_type_assistant(self, question_text, context, content):
        return self.client.chat.completions.create(
            model=self.gpt_model,
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

    # Инициализация экземпляра класса OpenAIService
    open_ai_service = OpenAIService()

    # Проверка метода user_type_assistant
    try:
        user_response = open_ai_service.user_type_assistant("Привет, как тебя зовут?")
        print("user_type_assistant работает корректно.")
        print(f"Ответ: {user_response}")
    except Exception as e:
        print(f"Ошибка в user_type_assistant: {e}")

    # Проверка метода developer_type_assistant
    try:
        developer_context = "Вы опытный разработчик Python."
        developer_response = open_ai_service.developer_type_assistant(
            question_text="Как создать виртуальное окружение в Python?",
            content=developer_context
        )
        print("developer_type_assistant работает корректно.")
        print(f"Ответ: {developer_response}")
    except Exception as e:
        print(f"Ошибка в developer_type_assistant: {e}")

    # Проверка метода assistant_type_assistant
    try:
        assistant_response = open_ai_service.assistant_type_assistant(
            question_text="Расскажи шутку.",
            context="",
            content=""
        )
        print("assistant_type_assistant работает корректно.")
        print(f"Ответ: {assistant_response}")
    except Exception as e:
        print(f"Ошибка в assistant_type_assistant: {e}")
