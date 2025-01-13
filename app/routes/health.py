from fastapi import APIRouter


health = APIRouter(prefix="/health")

@health.get("/")
def health_check():
    return {"status": "ok"}