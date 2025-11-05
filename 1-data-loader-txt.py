from langchain_community.document_loaders import TextLoader

loader = TextLoader("agentic_ai_sample.txt", encoding="utf-8")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")
print(documents)