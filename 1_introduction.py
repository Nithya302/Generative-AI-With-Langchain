from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI  
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatOpenAI(model="gpt-5-nano", temperature=0.0, max_tokens=1000)
# model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

response=model.invoke("What is the capital of India?")
print(response)
print(response.content)

