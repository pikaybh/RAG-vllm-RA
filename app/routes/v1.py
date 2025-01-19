from fastapi import APIRouter
from langserve import add_routes

from models import OpenAIModel, HuggingFaceModel
from chains import base_chains
from private import get_user
from structures import KrasRiskAssessmentInput


v1 = APIRouter(prefix="/v1")

yanolja = HuggingFaceModel(model_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

for chain in base_chains:
    setattr(yanolja, chain.__name__, chain)

add_routes(
    v1,
    yanolja.ra_chain(),
    path="/yanolja/ra",
    input_type=KrasRiskAssessmentInput
)

# openai_model = OpenAIModel(model_id="openai/gpt-4o")
# for chain in base_chains:
#     setattr(openai_model, chain.__name__, chain)
# 
# add_routes(
#     v1,
#     openai_model.ra_chain(),
#     path="/openai/ra",
#     input_type=KrasRiskAssessmentInput
# )

# add_routes(
#     v1,
#     model.ra_chain(method="at_list"),
#     path="/openai/ra/max"
# )
# 
# add_routes(
#     v1,
#     model.kras_chain(),
#     path="/openai/kras"
# )

__all__ = ["v1"]