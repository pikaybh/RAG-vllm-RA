import os
import sys
import logging

from fire import Fire
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Model Configuration
def create_rag_chain(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
        device_map="auto"
    )

    # Pipeline Configuration
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Embedding Configuration
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [
        {"content": "Python은 인기 있는 프로그래밍 언어입니다.", "metadata": {"source": "doc1"}},
        {"content": "Llama는 Meta에서 개발한 언어 모델입니다.", "metadata": {"source": "doc2"}},
    ]

    # Vector DB Configuration
    vector_store = FAISS.from_texts([text["content"] for text in texts], embedding_model)

    # RAG Chain
    retriever = vector_store.as_retriever()
    return RetrievalQA(llm=llm, retriever=retriever)

if __name__ == '__main__':
    rag_chain = Fire(create_rag_chain)
    query = ""
    while query != "exit":
        query = input("질문을 입력하세요: ")
        response = rag_chain.run(query)
        logger.info(response)
