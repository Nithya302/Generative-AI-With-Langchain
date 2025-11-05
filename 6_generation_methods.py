# from dotenv import load_dotenv
# from langchain_openai.chat_models import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from pydantic import BaseModel, Field
# import json

# load_dotenv() # load environment variables from .env file

# class Movie(BaseModel):
#     title: str = Field( description="The title of the movie")
#     director: str = Field(description="The director of the movie")
#     year: int = Field(description="The release year of the movie")

# # initialize the OpenAI chat model with output formatting
# model = ChatOpenAI(model="gpt-5-nano").with_structured_output(Movie)

# # Define a prompt template
# template = PromptTemplate.from_template(
#     """You are a helpful assistant that answers the details about the movie which the user asks.
#     Context: {context}
#     Question: {question}
#     Answer: """
# )

# # user_input = {
# #     "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010. It is a science fiction film that explores the concept of shared dreams and the manipulation of the subconscious.",
# #     "question": "What is the title, director, and year of release of the movie?"
# # }

# # Batch Demo
# batch_user_inputs=[{
#     "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010. It is a science fiction film that explores the context"
#     "question":"What is the title, director, and year of the release of the movie?"
# }]

# print("----------------------")
# # Batch Processing
# batch_response=chain.batch(batch_user_inputs)
# print("Batch Response:")
# for i, res in enumerate(batch_response):
#     print(f"Response {i+1}:")
#     print(res)
#     print("-----------------------")
#     # Convert the output to json
#     response_json=res.model_dump_json(indent=2)
#     print("Formatted JSON output:")
#     print(response_json) #print the json formatted response
    
# # Streaming demo
# print("Streaming Response:")
# print("---------------------")
# for token in chain.stream(batch_user_inputs):
#     print(token, end="", flush=True) #print the token as it is generated
    






from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()  # load environment variables from .env file

# -------------------- Initialize the OpenAI chat model --------------------
# Here we allow more creativity for generative responses
model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)  # higher temperature for creative generation

# -------------------- Define a generative prompt template --------------------
template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions based on the provided context.
If the context doesn't fully answer the question, generate a thoughtful, detailed response based on general knowledge.
Context: {context}
Question: {question}
Answer:"""
)

# -------------------- Create a chain --------------------
chain = LLMChain(
    llm=model,
    prompt=template
)

# -------------------- Example user input --------------------
user_input = {
    "context": "The capital of India is New Delhi. It is the seat of the government of India and is known for its rich history and cultural heritage.",
    "question": "Tell me about the capital of France and compare it briefly with New Delhi."
}

# -------------------- Generate response --------------------
response = chain.run(**user_input)

# -------------------- Output --------------------
print("--------------------------------")
print(response)
