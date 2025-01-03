import os

from openai import OpenAI

from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


OPENAI_API_KEY = ''

RAG_PROMPT_TEMPLATE = """
            You are a helpful coding assistant that can answer questions about the provided context.
            The context is usually a PDF document or an image (screenshot) of a code file.
            Augment your answers with code snippets from the context if necessary.
            
            If you don't know the answer, say you don't know.
            
            Context: {context}
            Question: {question}"""

PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


class OpenAIService:

    def __init__(self, gpt_model: str = 'gpt-4o'):

        self.gpt_model = gpt_model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))
        self.chat_client = ChatOpenAI(model=gpt_model, temperature=0)

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_assistant(self, question_text: str, retriever):
        """ Метод для работы с RAG-моделью:
            - Извлекает релевантные документы.
            - Формирует контекст.
            - Генерирует ответ с учётом контекста. """

        rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | PROMPT
                | self.chat_client
                | StrOutputParser()
        )

        return rag_chain.invoke(question_text)

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
