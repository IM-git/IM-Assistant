import os

from openai import OpenAI

from langchain_openai import ChatOpenAI


# В чем разница в использовании между hub и PromptTemplate для значения PROMPT?
''' 
Ответ:
    PromptTemplate:
    
        Что это такое?
            Это статически задаваемый шаблон, который создаётся прямо в коде. Вы можете жёстко задать структуру запроса.
        
        Когда использовать?
            Если структура вашего запроса не изменяется и не требует динамической загрузки.
        
        Пример:
            from langchain.prompts import PromptTemplate
            
            RAG_PROMPT_TEMPLATE = """
            Context: {context}
            Question: {question}
            """
            PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            
            # Использование
            prompt_text = PROMPT.format(context="Python — это язык программирования.", question="Что такое Python?")
            print(prompt_text)
            
    hub.pull:
    
        Что это такое?
            Это динамически загружаемый шаблон из LangChain Hub — централизованного хранилища для шаблонов,
            цепочек и других компонентов. Вы можете использовать готовые шаблоны, предоставляемые LangChain,
            или загружать пользовательские шаблоны, созданные вами.
            
        Когда использовать?
            Если вы хотите динамически загружать шаблоны, поддерживать их версионность
            или использовать шаблоны, предоставляемые сообществом.
            
        Пример:
            from langchain import hub
            
            # Загрузка шаблона с LangChain Hub
            PROMPT = hub.pull("rlm/rag-prompt")
            
            # Использование
            prompt_text = PROMPT.format(context="Python — это язык программирования.", question="Что такое Python?")
            print(prompt_text)
'''
from langchain import hub
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
# PROMPT = hub.pull("rlm/rag-prompt")


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
