from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_doc, test_split, download_embedding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key = os.environ.get("pinecone_api_key")

os.environ["pinecone_api_key"] = pinecone_api_key

extracted_data = load_pdf_file(data="data/")
filtered_data = filter_doc(docs=extracted_data)
chunk_data = test_split(minimal_docs=filtered_data)

embeddings = download_embedding()

pinecone_key = pinecone_api_key
pc = Pinecone(api_key=pinecone_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
doc = PineconeVectorStore.from_documents(
    documents=chunk_data,
    embedding=embeddings,
    index_name=index_name
)