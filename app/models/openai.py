import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from models.base import BaseLanguageModel
from utils import dantic2json


load_dotenv()



class OpenAIModel(BaseLanguageModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)

        # Secret Key
        __sk = os.getenv("OPENAI_API_KEY")

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
    model = OpenAIModel(model_id="openai/gpt-4o")
    # ra_chain = model.ra_chain()
    ra_chain = model.ra_chain()
    response = ra_chain.invoke(
        {
            "input": "철근 배근 작업", 
            # "number": 10
        }
    )

    # 결과 출력
    response_json = dantic2json(response)
    print(response_json)
