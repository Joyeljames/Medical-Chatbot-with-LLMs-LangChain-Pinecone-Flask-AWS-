from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import torch


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    Documents = loader.load()
    return Documents

#filtering the document removed unwanted things and keeps useful data its a cleaning process
def filter_doc(docs: List[Document]) -> List[Document]:
    filtered_docs: List[Document] = []
    for doc in docs:
        stc = doc.metadata.get("source")
        filtered_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": stc}
            )
        )
    return filtered_docs



# splitting the document into smaller chunks
def test_split(minimal_docs):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunk = text_spliter.split_documents(minimal_docs)
    return chunk

#downloading the embedding model and creating the embedding object
def download_embedding():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                                       )
    return embeddings

