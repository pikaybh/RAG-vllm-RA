import os
import yaml
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union, Iterator

# 현재 파일의 위치에서 한 단계 상위 폴더로 설정
SELF_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 경로
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
BASE_DIR = os.path.join("templates", "prompts")


def convert2chat(raw_data: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert raw prompt data into a chat-compatible format.

    Args:
        raw_data (Dict[str, Dict[str, str]]): The raw prompt data.

    Returns:
        Dict[str, List[Dict[str, str]]]: Chat-compatible prompt data.
    """
    return {k: [{"role": role, "content": content} for role, content in v.items()] for k, v in raw_data.items()}


def convert2msg(raw_data: Dict[str, Dict[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Convert raw prompt data into a chat-compatible format.

    Args:
        raw_data (Dict[str, Dict[str, str]]): The raw prompt data.

    Returns:
        Dict[str, List[Dict[str, str]]]: Chat-compatible prompt data.
    """
    return {k: [(role, content) for role, content in v.items()] for k, v in raw_data.items()}



class PromptBuilder:
    def __init__(self, task: str, ptype: Optional[str] = "message"):
        """
        Initialize the PromptBuilder with a specific task and data format.

        Args:
            task (str): The name of the task, corresponding to the YAML template file name.
            ptype (Optional[str]): The format type for the prompts. Options are "message" or "chat".

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            ValueError: If an invalid `ptype` is provided.
        """
        self.task = task
        self.file_path = os.path.join(ROOT_DIR, BASE_DIR, f"{task}.yaml")

        # Check if the YAML file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The specified YAML file '{self.file_path}' does not exist.")

        # Load the YAML data
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)

        # Validate the ptype argument
        if ptype not in {"message", "chat"}:
            raise ValueError(f"Invalid ptype '{ptype}'. Expected 'message' or 'chat'.")

        # Convert prompts based on the ptype
        if ptype == "chat":
            self.prompts = convert2chat(self.raw)
        elif ptype == "message":
            self.prompts = convert2msg(self.raw)
        else: 
            self.prompts = self.raw

    def __getitem__(self, method: str) -> Dict[str, str]:
        """
        Retrieve a specific method from the prompts.

        Args:
            method (str): The method to retrieve.

        Returns:
            Dict[str, str]: The prompt corresponding to the specified method.

        Raises:
            KeyError: If the specified method does not exist.
        """
        try:
            return self.prompts[method]
        except KeyError:
            raise KeyError(f"Step '{method}' not found in the prompts for task '{self.task}'.")

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, str]]]:
        """
        Make the PromptBuilder iterable over its methods.

        Returns:
            Iterator[Tuple[str, Dict[str, str]]]: An iterator over the methods and their prompts.
        """
        return iter(self.prompts.items())

    def get_methods(self, methods: Optional[Union[str, List[str]]] = None) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Retrieve specific methods or all prompts.

        Args:
            methods (Optional[Union[str, List[str]]]): Specific methods to extract. If None, all prompts are returned.

        Returns:
            Dict[str, str] | List[Dict[str, str]]: The prompts corresponding to the requested methods.

        Raises:
            KeyError: If a requested method does not exist.
            TypeError: If the `methods` parameter is neither a string, a list of strings, nor None.
        """
        if not methods:
            return self.prompts

        if isinstance(methods, list):
            try:
                return [self.prompts[method] for method in methods]
            except KeyError as e:
                raise KeyError(f"Step '{e.args[0]}' not found in the prompts for task '{self.task}'.")

        if isinstance(methods, str):
            try:
                return self.prompts[methods]
            except KeyError:
                raise KeyError(f"Step '{methods}' not found in the prompts for task '{self.task}'.")

        raise TypeError("The 'methods' parameter must be a string, a list of strings, or None.")


__all__ = ["PromptBuilder"]
