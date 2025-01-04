import json


class FileTools:

    @staticmethod
    def read_file(file_path: str) -> list | dict | tuple:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
