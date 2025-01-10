import fire

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