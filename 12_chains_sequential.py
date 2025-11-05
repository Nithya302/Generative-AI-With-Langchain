from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model=ChatOpenAI(model="gpt-4o")

# Define prompt templates
animal_facts_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Define a prompt template for translatioin to French
translation_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
count_words=RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation=RunnableLambda(lambda output: {"text": output, "language":"french"})

# create the combined chain using langchain expression language (lcel)
chain=animal_facts_template | model| StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() | count_words

# Run the chain
result=chain.invoke({"animal":"cat", "count":2})

# Output
print(result)