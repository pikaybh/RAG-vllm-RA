import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from models import BaseLanguageModel
from utils import dantic2json, get_logger, timer

logger = get_logger("models.huggingface")
load_dotenv()


class HuggingFaceModel(BaseLanguageModel):
    def __init__(
        self, 
        model_id: str, 
        logger: Optional[logging.Logger] = logger, 
        api_key: Optional[str] = None
    ):
        """Initialize HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID
            logger: Optional logger instance
            api_key: Optional HuggingFace API key
        """
        super().__init__(model_id, logger)

        # Secret Key
        # __sk = os.getenv("HUGGINGFACE_API_KEY") or api_key

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            # token=__sk,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            device_map="auto"  # 자동으로 적절한 디바이스 선택
        )
        
        # Initialize LangChain pipeline
        self.model = HuggingFacePipeline(pipeline=pipe)
        
        # Initialize Embedding Model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_id,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )


__all__ = ["HuggingFaceModel"]

if __name__ == "__main__":
    logger.info("Starting risk assessment process...")
    
    @timer
    def run_assessment():
        model = HuggingFaceModel(model_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
        payload = {
            "work_type": "철근 작업",
            "procedure": "철근 가공 및 운반"
        }
        ra_chain = model.silent_ra_chain()
        response = ra_chain.invoke(payload)
        return dantic2json(response)

    response_json = run_assessment()
    logger.info(f"Final response: {response_json}")

