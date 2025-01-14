import json

def dantic2json(data) -> json:
    # data를 Pydantic 모델로 가정
    data_dict = data.model_dump()  # Pydantic 모델을 Python 딕셔너리로 변환
    return json.dumps(data_dict, indent=4, ensure_ascii=False)  # JSON 문자열로 변환

__all__ = ["dantic2json"]