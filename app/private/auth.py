import secrets
import logging
from functools import wraps

from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_core.runnables import Runnable

from private.db import check_api_key, get_user_from_api_key


# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

api_key_header = APIKeyHeader(name="X-API-Key")

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, os.getenv('BEDROCK_CHAT_USERNAME', 'bedrock'))
    correct_password = secrets.compare_digest(
        credentials.password, os.getenv('BEDROCK_CHAT_PASSWORD', 'bedrock'))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_user(api_key_header: str = Security(api_key_header)):
    if check_api_key(api_key_header):
        user = get_user_from_api_key(api_key_header)
        print(user)
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key"
    )


__all__ = ["get_user"]

"""
def get_user(func):
    "" "
    데코레이터로 API 키를 검증하고 사용자 정보를 함수에 주입.
    "" "
    @wraps(func)
    def wrapper(*args, api_key_header: str = Security(api_key_header), **kwargs):
        if not check_api_key(api_key_header):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key"
            )
        user = get_user_from_api_key(api_key_header)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid user"
            )
        # 함수에 사용자 정보를 전달
        return func(*args, **kwargs)  #  user=user,
    return wrapper

class SecureChainWrapper(Runnable):
    def __init__(self, chain, api_key_name="X-API-Key"):
        self.chain = chain
        self.api_key_name = api_key_name

    def verify_user(self, api_key):
        logger.debug(f"Verifying API key: {api_key}")
        if not check_api_key(api_key):
            logger.warning("Invalid API key attempted")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key"
            )
        user = get_user_from_api_key(api_key)
        if not user:
            logger.warning(f"API key {api_key} is valid but user is invalid")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid user"
            )
        logger.info(f"User {user['name']} authenticated successfully")
        return user

    async def invoke(self, inputs, api_key=None):
        # 인증 및 사용자 검증
        if not api_key:
            logger.error("No API key provided in the request")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key '{self.api_key_name}' is missing"
            )
        user = self.verify_user(api_key)
        logger.info(f"Authenticated user: {user['name']} ({user['id']})")

        # 체인 실행
        logger.debug("Invoking chain with inputs.")
        return await self.chain.invoke(inputs)

    async def __call__(self, inputs, api_key=None):
        return await self.invoke(inputs, api_key)

"""