import os
import yaml
from typing import Dict, List, Optional, Tuple

# 현재 파일의 위치에서 한 단계 상위 폴더로 설정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 경로
ROOT_DIR = os.path.dirname(ROOT_DIR)  # 한 단계 상위 폴더로 이동
BASE_DIR = "templates"

def build_prompt(task: str, steps: Optional[str | List[str]] = None) -> Dict[str, str] | List[Dict[str, str]]:
    """
    Construct prompts based on a YAML template file.

    Args:
        task (str): The name of the task, corresponding to the YAML template file name.
        steps (Optional[str | List[str]]): Specific steps to extract from the loaded prompts. If a string is provided,
            it is interpreted as a single step. If a list of strings is provided, multiple steps are extracted.

    Returns:
        Dict[str, str] | List[Dict[str, str]]: The prompts corresponding to the requested steps. If `steps` is None,
        the entire prompts dictionary is returned.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        KeyError: If a requested step does not exist in the prompts dictionary.
        TypeError: If the `steps` parameter is neither a string, a list of strings, nor None.
    """

    # Construct the file path for the YAML file.
    file_path = os.path.join(ROOT_DIR, BASE_DIR, f"{task}.yaml")

    # Check if the file exists before attempting to open it.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified YAML file '{file_path}' does not exist.")
    print(file_path)
    # Load prompts from the YAML file.
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    if not steps:
        return prompts

    # Handle steps if it is a list.
    if isinstance(steps, list):
        try:
            return [prompts[step] for step in steps]
        except KeyError as e:
            raise KeyError(f"Step '{e.args[0]}' not found in the prompts for task '{task}'.")

    # Handle steps if it is a string.
    if isinstance(steps, str):
        try:
            return prompts[steps]
        except KeyError:
            raise KeyError(f"Step '{steps}' not found in the prompts for task '{task}'.")

    # Raise TypeError if steps is of an unsupported type.
    raise TypeError("The 'steps' parameter must be a string, a list of strings, or None.")


def payloader(*args) -> List[Tuple[str, str]]:
    """TODO: 고쳐야 함!!! It's temporal"""
    return [(k, v) for k, v in enumerate(args)]

__all__ = ["build_prompt", "payloader"]