from pydantic import BaseModel, Field
from typing import List, Literal


class RiskItem(BaseModel):
    번호: int = Field(description="시리얼 숫자")
    위험요소: str = Field(description="식별된 위험 요소")
    위험성평가: str = Field(description="해당 위험 요소의 위험성 평가 결과")
    위험도: Literal["높음", "중간", "낮음"] = Field(description="해당 위험 요소의 위험도")
    감소대책: List[str] = Field(description="위험 요소 감소를 위해 권장되는 통제 및 제한 조치 목록")



class RiskAssessmentOutput(BaseModel):
    작업: str = Field(description="작업의 이름")
    위험성평가표: List[RiskItem] = Field(description="각 위험 요소에 대한 위험성 평가와 통제 조치 목록")
    기타: List[str] = Field(description="기타 제언")


__all__ = ["RiskAssessmentOutput"]