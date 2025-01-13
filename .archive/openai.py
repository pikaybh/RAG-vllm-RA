from fastapi import APIRouter
from langserve import add_routes


openai = APIRouter(prefix="/openai")

add_routes(
    openai,
    ChatOpenAI(
        model="gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    path="/openai",
)

@openai.post("/chat")
def chat(model: str, prompt: str):
    return openai.chat(model, prompt)