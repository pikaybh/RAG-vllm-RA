import os
from typing import List, Literal
from pydantic import BaseModel, Field

from utils import get_unique


# 현재 파일의 위치에서 두 단계 상위 폴더로 설정
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DIR = "assets"

CLS = ("공종", "공정", "공정설명", "설비", "물질", "유해위험요인 분류", "유해위험요인 원인", "위험 가능성", "위험 중대성", "위험성")
FILE_NAME = "KRAS_찬영수정_250114.csv"
T_DICT = get_unique(os.path.join(ROOT_DIR, BASE_DIR, FILE_NAME), *CLS, encoding="utf-8")



class KrasWorkInput(BaseModel):
    work_type: str = Field(description="작업 공종의 이름")
    procedure: str = Field(description="작업 공정의 이름")



class KrasRiskAssessmentInput(BaseModel):
    input: KrasWorkInput



"""
class RiskItem(BaseModel):
    번호: int = Field(description="시리얼 숫자")
    공종: Literal[*T_DICT["공종"]] = Field(description="작업 공종의 이름")
    공정: Literal[*T_DICT["공정"]] = Field(description="작업 공정의 이름")
    공정설명: Literal[*T_DICT["공정설명"]] = Field(description="작업 공정 설명")
    설비: Literal[*T_DICT["설비"]] = Field(description="작업에 사용되는 설비 이름")
    물질: Literal[*T_DICT["물질"]] = Field(description="작업 과정에서 취급되는 물질 이름")
    유해위험요인_분류: Literal[*T_DICT["유해위험요인 분류"]] = Field(description="유해 또는 위험 요인의 분류")
    유해위험요인_원인: Literal[*T_DICT["유해위험요인 원인"]] = Field(description="유해 또는 위험 요인의 발생 원인")
    유해위험요인: str = Field(description="유해 또는 위험 요인의 상세")
    관련근거: str = Field(description="관련된 근거 법령")
    위험_가능성: Literal[*T_DICT["위험 가능성"]] = Field(description="위험이 발생할 가능성")
    위험_중대성: Literal[*T_DICT["위험 중대성"]] = Field(description="위험이 미치는 영향의 심각성")
    위험성: Literal[*T_DICT["위험성"]] = Field(description="해당 위험 요소의 위험도")
    감소대책: List[str] | str = Field(description="위험 요소 감소를 위해 권장되는 통제 및 제한 조치 목록")
"""
class RiskItem(BaseModel):
    번호: int = Field(description="시리얼 숫자")
    공종: str = Field(description="작업 공종의 이름")
    공정: str = Field(description="작업 공정의 이름")
    공정설명: str = Field(description="작업 공정 설명")
    설비: str = Field(description="작업에 사용되는 설비 이름")
    물질: str = Field(description="작업 과정에서 취급되는 물질 이름")
    유해위험요인_분류: str = Field(description="유해 또는 위험 요인의 분류")
    유해위험요인_원인: str = Field(description="유해 또는 위험 요인의 발생 원인")
    유해위험요인: str = Field(description="유해 또는 위험 요인의 상세")
    관련근거: str = Field(description="관련된 근거 법령")
    위험_가능성: str = Field(description="위험이 발생할 가능성")
    위험_중대성: str = Field(description="위험이 미치는 영향의 심각성")
    위험성: str = Field(description="해당 위험 요소의 위험도")
    감소대책: List[str] | str = Field(description="위험 요소 감소를 위해 권장되는 통제 및 제한 조치 목록")




class KrasRiskAssessmentOutput(BaseModel):
    공종: str = Field(description="사용자가 입력한 공종의 이름")
    공정: str = Field(description="사용자가 입력한 공정의 이름")
    작업명: str = Field(description="사용자가 입력한 작업명")
    위험성평가표: List[RiskItem] = Field(description="각 위험 요소에 대한 위험성 평가와 통제 조치 목록")
    기타: List[str] = Field(description="기타 제언")


__all__ = ["KrasRiskAssessmentInput", "KrasRiskAssessmentOutput"]
