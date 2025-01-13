from fastapi import APIRouter
from langserve import add_routes

llama = APIRouter(prefix="/llama")

add_routes(
    llama,
    ChatOpenAI(
        model="gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    path="/openai",
)


