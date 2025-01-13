from fastapi import APIRouter
from langserve import add_routes

from models.llama import LlamaModel
from models.openai import OpenAIModel


v1 = APIRouter(prefix="/v1")

# yanolja = HF(model_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
# eeve_rag_chain = yanolja.risk_assessment()
# 
# add_routes(
#     v1,
#     eeve_rag_chain,
#     path="/yanolja/ra"
# )

model = OpenAIModel(model_id="openai/gpt-4o")
gpt_rag_chain = model.risk_assessment()

add_routes(
    v1,
    gpt_rag_chain,
    path="/openai/ra"
)