from typing import List
from pydantic import BaseModel, Field



class BaseDocuments(BaseModel):
    name: str = None
    manual: List[str] = ["2023년도 위험성평가 및 안전보건관리체계 우수사례집.pdf"]
    reference: List[str] = ["KRAS_찬영수정_250114.csv"]
    regulation: List[str] = ["안전보건규칙.txt"]  # ["안전보건규칙_수정본(250120).json"]


__all__ = ["BaseDocuments"]