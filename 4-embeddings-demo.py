from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv() # load environment variables from .env file

model = OpenAIEmbeddings(model="text-embedding-3-small")  #small and /large are available
#model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Example text to embed
text = ["The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "LangChain is a framework for building applications with LLMs.",
        "Embeddings are numerical representations of text that capture semantic meaning."]

# Generate embeddings for the text
embeddings = model.embed_documents(text)
print("Embeddings:")

for i, emb in enumerate(embeddings):
    print(f"Total dimension in each embedding vector: {len(emb)} Text {i+1}: {emb[:10]}...")  # Print first 10 dimensions of the embedding for brevity
