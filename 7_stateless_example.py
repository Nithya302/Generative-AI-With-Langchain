from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv() # load environment variables from .env file

# initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-5-nano")

user_input=["Kanthraju has won best student in my class",
            "Who won best student in my class?",]

response=model.batch(user_input)

print("Batch Response:")
for i, res in enumerate(response):
    print(f"Response {i+1}:")
    print(res.content)
    print("------------------------")