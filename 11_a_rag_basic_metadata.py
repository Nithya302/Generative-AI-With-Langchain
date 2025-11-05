import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "E:\AI syllabus\langchain\RAG")
db_dir = os.path.join(current_dir, "E:\AI syllabus\langchain\RAG\db")
persistent_directory = os.path.join(db_dir, "E:\AI syllabus\langchain\RAG\chroma_db")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()

        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"\nTotal chunks created: {len(docs)}")

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create and persist the Chroma DB
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("✅ Vector store created and persisted with metadata!")

else:
    print("✅ Vector store already exists. No need to initialize.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Example query
    query = "Where does Gandalf meet Frodo?"
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    results = retriever.invoke(query)

    print("\n--- Query Results ---")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: {doc.page_content}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}\n")