# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# import json

# load_dotenv()  # load environment variables from .env

# # -------------------- Define Pydantic model for structured output --------------------
# class Movie(BaseModel):
#     title: str = Field(description="The title of the movie")
#     director: str = Field(description="The director of the movie")
#     year: int = Field(description="The release year of the movie")
#     description: str = Field(description="Brief description or summary of the movie")  # added generative field

# # -------------------- Initialize the chat model --------------------
# # Generative mode with higher temperature
# model = ChatOpenAI(model="gpt-5-nano",temperature=1)
# parser = PydanticOutputParser(pydantic_object=Movie)

# # -------------------- Define generative prompt template --------------------
# template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a helpful assistant that answers questions about movies.
# Use the context if available, but if the context does not fully answer, generate additional details based on general knowledge.
# Context: {context}
# Question: {question}
# Answer in structured format matching the Pydantic model (title, director, year, description):
# """
# )

# # -------------------- Create LLMChain --------------------
# chain = LLMChain(
#     llm=model,
#     prompt=template,
#     output_parser=parser
# )

# # -------------------- Example input --------------------
# user_input = {
#     "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010.",
#     "question": "Give me the full details of the movie including a short summary."
# }

# # -------------------- Generate response --------------------
# response = chain.run(user_input)

# # -------------------- Parse structured output --------------------
# movie_struct = parser.parse(response)

# # -------------------- Print results --------------------
# print("Raw Response:\n", response)
# print("\nStructured JSON Output:\n", movie_struct.model_dump_json(indent=2))
















from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()  # Load environment variables from .env

# -------------------- Define Pydantic model for structured output --------------------
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The release year of the movie")
    description: str = Field(description="Brief description or summary of the movie")

# -------------------- Initialize the chat model --------------------
# gpt-5-nano only supports temperature=1
model = ChatOpenAI(model="gpt-5-nano", temperature=1)

# -------------------- Create output parser --------------------
parser = PydanticOutputParser(pydantic_object=Movie)

# -------------------- Define prompt template --------------------
template = PromptTemplate(
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are a helpful assistant that answers questions about movies.

Use the context if available, but if the context does not fully answer, generate additional details based on general knowledge.

Context: {context}
Question: {question}

Answer in JSON format:
{format_instructions}
"""
)

# -------------------- Build the chain using the new RunnableSequence style --------------------
chain = template | model | parser

# -------------------- Example input --------------------
user_input = {
    "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010.",
    "question": "Give me the full details of the movie including a short summary."
}

# -------------------- Generate response --------------------
movie_struct = chain.invoke(user_input)

# -------------------- Print results --------------------
print("\nStructured JSON Output:\n", movie_struct.model_dump_json(indent=2))
