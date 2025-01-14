import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from models.base import BaseLanguageModel
from utils import dantic2json



class HuggingFaceModel(BaseLanguageModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)

        # Initialize Language Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # Initialize Embedding Model
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    def ra_chain(self):
        """
        Placeholder method for risk assessment chain.
        Replace with actual implementation as needed.
        """
        def invoke(input_data):
            input_text = input_data.get("input", "")
            return self.pipeline(input_text, max_length=100, num_return_sequences=1)

        return type("RAChain", (object,), {"invoke": invoke})()



__all__ = ["HuggingFaceModel"]

if __name__ == "__main__":
    model = HuggingFaceModel(model_id="gpt2")  # Replace with the desired Hugging Face model ID
    ra_chain = model.ra_chain()
    response = ra_chain.invoke(
        {
            "input": "철근 배근 작업",
        }
    )

    # 결과 출력
    response_json = dantic2json(response)
    print(response_json)