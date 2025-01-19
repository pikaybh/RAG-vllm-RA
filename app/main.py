"""
LangChain Server
Version: 1.0
Author: @pikaybh
Copyright (c) 2025 SNUCEM. All rights reserved.

Description:
    This API server is designed to provide a robust and scalable interface utilizing LangChain's Runnable interfaces.
"""

from fastapi import APIRouter, FastAPI, Depends
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from routes import secure, public
from private import get_user


# API Configuration
app = FastAPI(
    title="LangChain Server",
    version="1.0.0",
    description="A simple api server using Langchain's Runnable interfaces",
    dependencies=[Depends(get_user)]
)

# Fouter Configuration
router = APIRouter()

# Redirects
@router.get("/", include_in_schema=False)
def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Include routes
app.include_router(router)
for route in public:
    app.include_router(route)
for route in secure:
    app.include_router(route, dependencies=[Depends(get_user)])

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["*"]
)

"""
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
"""
