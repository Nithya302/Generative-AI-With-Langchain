# PyPDFParser → A parser designed to read PDF files page by page.
# Blob → Represents the file data (binary/text).

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob

# Now, instead of passing raw bytes or file streams, you pass blob to the parser.
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

parser = PyPDFParser() # Initialize the parser which know extract text + metadata from PDF files.

documents = parser.lazy_parse(blob)
docs = []
for doc in documents:
    docs.append(doc)
# print(docs[0].page_content)
# print(docs[0].metadata)
print(docs)