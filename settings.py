import os
import dotenv


dotenv.load_dotenv('.env')
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

gpt_model = ""  # get models list [print(_["id"]) for _ in openai.Model.list()["data"]]
