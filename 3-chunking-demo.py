from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

parser = PyPDFParser()

documents = parser.lazy_parse(blob)
docs = []
for doc in documents:
    docs.append(doc)
# print(docs[0].page_content)
# print(docs[0].metadata)
print(docs)

# Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Adjust chunk size as needed
    chunk_overlap=50  # Adjust overlap as needed
)

chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)  # Print the content of each chunk
    print("--------------------------------")