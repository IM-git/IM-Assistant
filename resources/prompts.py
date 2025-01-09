class Prompts:

    RAG_PROMPT_TEMPLATE = """
    You are a helpful coding assistant that can answer questions about the provided context.
    The context is usually a PDF document or an image (screenshot) of a code file.
    Augment your answers with code snippets from the context if necessary.

    If you don't know the answer, say you don't know.

    Context: {context}
    Question: {question}
    """

    # Путь к векторному хранилищу для обработанного PDF
    PDF_VECTORSTORE_PATH = "./vectorstore/processed_pdf"

    # Путь к векторному хранилищу для истории вопросов-ответов
    HISTORY_VECTORSTORE_PATH = "./vectorstore/qa_history"


