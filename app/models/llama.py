from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

from models.base import Agent
from utils import build_prompt



class LlamaModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation"
        )

    def risk_assessment(self) -> object:
        self.prompt = ChatPromptTemplate.from_template(
            build_prompt(
                organization=self.organization, 
                tasks="risk_assessment"
            )
        )
        
        return self.prompt | self.model
