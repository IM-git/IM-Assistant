from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from tools.file_tools import FileTools
from resources import Prompts

PROMPT = PromptTemplate.from_template(Prompts.RAG_PROMPT_TEMPLATE)


class LangChainService:

    def __init__(self, gpt_model: str = 'gpt-4o-mini'):

        self.chat_client = ChatOpenAI(model=gpt_model, temperature=0)
        self.format_docs = FileTools().format_docs

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

