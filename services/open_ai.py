import os

from openai import OpenAI

OPENAI_API_KEY = ''


class OpenAIService:

    def __init__(self, gpt_model: str = 'gpt-4o-mini'):

        self.gpt_model = gpt_model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

    def user_type_assistant(self, question_text: str):
        """Функция для обработки запроса от пользователя"""
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
