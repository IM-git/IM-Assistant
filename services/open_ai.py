import os
from openai import OpenAI

OPENAI_API_KEY = ''
# client = OpenAI()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not setd as env var>"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

GPT_MODEL = 'gpt-4o'


def user_type_assistant(question_text: str):
    return client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": question_text
                }
            ]
        )

def developer_type_assistant(question_text: str, content: str):
    return client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "developer",
                 "content": [
                     {
                         "type": "text",
                         "text":
                             "You are a helpful assistant that answers programming"
                             "questions in the style of a southern belle from the "
                             "southeast United States."
                     }
]
                 },
                {"role": "user",
                 "content": question_text
                 }
            ]
        )

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
        )


if __name__ == '__main__':

    _user_assistant_question = "Write a haiku about recursion in programming."

    _developer_assistant_question = "Чем питаются вороны?"
    # _developer_assistant_question = input('Привет! Напишите свой вопрос: ')
    _developer_assistant_context = ("You are a helpful assistant that answers programming questions "
                                    "in the style of a southern belle from the southeast United States.")

    _assistant_question = "Write a haiku about recursion in programming."
    _assistant_context = ""
    _assistant_content = ""

    # _user_assistant = user_type_assistant(question_text=_user_assistant_question).choices[0].message
    # _developer_assistant = developer_type_assistant(question_text=_developer_assistant_question,
    #                                                 content=_developer_assistant_context).choices[0].message
    # _assistant = assistant_type_assistant(question_text=_assistant_question,
    #                                       context=_assistant_context,
    #                                       content=_assistant_content).choices[0].message

    # print(dict(_developer_assistant)['content'])
    while True:
        _developer_assistant_question = input('Привет! Напишите свой вопрос: ')
        _developer_assistant = developer_type_assistant(question_text=_developer_assistant_question,
                                                        content=_developer_assistant_context).choices[0].message
        print(dict(_developer_assistant)['content'])

