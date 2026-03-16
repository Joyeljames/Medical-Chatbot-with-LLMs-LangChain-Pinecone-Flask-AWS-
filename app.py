from flask import Flask, request, jsonify,render_template
from langchain import HuggingFacePipeline
from torch import embedding
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.prompts import *
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.helper import download_embedding

app = Flask(__name__)


load_dotenv()

pinecone_api_key = os.environ.get("pinecone_api_key")

os.environ["pinecone_api_key"] = pinecone_api_key

# This for searching the document in the already created index
index_name = "medical-chatbot"
from langchain_pinecone import PineconeVectorStore


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=download_embedding()

)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k": 1})

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,
    temperature=0.7,
    return_full_text=False,
    do_sample = True
)

chatmodel = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:")
])

question_answer_chain = create_stuff_documents_chain(chatmodel,prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")



@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    try:
        # RAG response
        response = rag_chain.invoke({"input": msg})
        # Check which key exists
        text = response.get("answer") or response.get("output_text") or str(response)
    except Exception as e:
        text = f"Error: {str(e)}"
    print("User:", msg)
    print("Bot:", text)
    return text


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)