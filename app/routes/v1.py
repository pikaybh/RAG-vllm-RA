from fastapi import APIRouter
from langserve import add_routes

from models import (OpenAIModel,
                    HuggingFaceModel)
from private import get_user
from templates import KrasRiskAssessmentInput


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

# add_routes(
#     v1,
#     model.ra_chain(method="at_list"),
#     path="/openai/ra/max"
# )
# 
# add_routes(
#     v1,
#     model.full_chain(),
#     path="/openai/kras"
# )

add_routes(
    v1,
    model.kras_chain(),
    path="/openai/kras",
    input_type=KrasRiskAssessmentInput
)

__all__ = ["v1"]