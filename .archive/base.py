import os


# 현재 파일의 위치에서 한 단계 상위 폴더로 설정
SELF_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 경로
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
BASE_DIR = os.path.join("assets", "structures")

def get_addr(file_name: str) -> str:
    """
    Args:
        file_name(str): It should contain extension

    Return:
        str: full path
    """
    return os.path.join(ROOT_DIR, BASE_DIR, file_name)

__all__ = ["get_addr"]

print(os.path.join(ROOT_DIR, BASE_DIR))