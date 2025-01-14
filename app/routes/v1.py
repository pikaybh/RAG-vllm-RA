from fastapi import APIRouter, Query
from langserve import add_routes

from models import (OpenAIModel,
                    HuggingFaceModel)


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

add_routes(
    v1,
    model.ra_chain(),
    path="/openai/ra"
)

add_routes(
    v1,
    model.ra_chain(method="at_list"),
    path="/openai/ra/max"
)