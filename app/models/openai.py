import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from chains import base_chains
from models import BaseLanguageModel
from utils import dantic2json, get_logger, timer, add_chain


logger = get_logger("models.openai")
load_dotenv()



class OpenAIModel(BaseLanguageModel):
    def __init__(self, model_id: str, logger: Optional[logging.Logger] = logger, api_key: Optional[str] = None):
        super().__init__(model_id, logger)

        # Secret Key
        __sk = os.getenv("OPENAI_API_KEY") or api_key

        # Initialize Engine
        self.model = ChatOpenAI(
            model=self.model_name, 
            api_key=__sk
        )

        # Initialize Embedding Model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            api_key=__sk
        )



__all__ = ["OpenAIModel"]

if __name__ == "__main__":
    logger.info("Starting risk assessment process...")
    
    @timer
    def run_assessment():
        # Initialize Model
        model = OpenAIModel(model_id="openai/gpt-4o")

        # Add Chains
        for chain in base_chains:
            add_chain(model, chain)
        
        # Input Data
        payload = {
            "work_type": "철근 작업",
            "procedure": "철근 가공 및 운반"
        }

        # Initialize Chain
        # ra_chain = model.ra_chain()
        ra_chain = model.cl_chain()

        # Run Chain
        response = ra_chain.invoke(payload)

        return dantic2json(response)

    response_json = run_assessment()
    logger.info(f"Final response: {response_json}")

