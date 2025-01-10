import os

import fire
import torch
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = Flask(__name__)


class RetrievalChain:
    def __init__(
        self,
        embedding_model_name: str,
        llm_model_name: str,
        llm_host: str,
        llm_api_key: str,
        chroma_collection_name: str = "rag-chatbot",
        persist_directory: str = "./chroma_store"
    ):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        
        self.vector_store = Chroma(
            chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        self.llm = ChatOpenAI(
            base_url=llm_host,
            model=llm_model_name,
            openai_api_key=llm_api_key,
            temperature=0.5,
            max_tokens=4096,
        )

        self.prompt = ChatPromptTemplate.from_template(
            "As an AI assistant, you are tasked with answering questions based on provided context.\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:")

        self.chain = RunnableParallel(
            {
                "context": self.vector_store.as_retriever(),
                "question": RunnablePassthrough(),
            }
        ).assign(
            answer=(
                RunnablePassthrough().assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        )

    def handle_request(self, message: str):
        try:
            response = self.chain.invoke({"question": message})
            return response["answer"]
        except Exception as e:
            return str(e)

    def add_pdf_to_db(self, file_path):
        try:
            loader = PyMuPDFLoader(file_path=file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
            chunks = text_splitter.split_documents(documents)

            self.vector_store.add_documents(chunks)
            self.vector_store.persist()

            return {"status": "success", "message": f"Added {len(chunks)} chunks from {file_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    message = data["message"]
    answer = retrieval_chain.handle_request(message)
    return jsonify({"answer": answer})


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)

    result = retrieval_chain.add_pdf_to_db(file_path)
    os.remove(file_path)

    return jsonify(result)


# Configuration
def start_server(
    embedding_model_name="BAAI/bge-m3",
    llm_model_name="casperhansen/llama-3-8b-instruct-awq",
    llm_host="http://localhost:8000/v1",
    llm_api_key="your-api-key",
    port=8000,
    persist_directory="./chroma_store"
):
    global retrieval_chain
    retrieval_chain = RetrievalChain(
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        llm_host=llm_host,
        llm_api_key=llm_api_key,
        persist_directory=persist_directory
    )

    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    fire.Fire(start_server)
