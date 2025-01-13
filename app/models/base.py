"""TODO: 고민 좀 해보기..."""

import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore


ROOT_DIR = os.getenv("ROOT_DIR", "")
BASE_DIR = "assets"
FILE_FORMATS = (".pdf",)



class BaseModel:
    """Represents an agent for loading and managing document loaders.

    Attributes:
        organization (str): The organization associated with the agent.
        model_name (str): The model name for the agent.
        _loaders (List[PyPDFLoader]): Internal list of document loaders.
    """

    def __init__(self, model_id: str):
        """Initializes the Agent with a model identifier and loads documents.

        Args:
            model_id (str): The model identifier in the format 'organization/model_name'.

        Raises:
            ValueError: If the model_id does not contain a '/' to split organization and model name.
        """
        try:
            self.organization, self.model_name = model_id.split("/")
        except ValueError:
            raise ValueError("model_id must be in the format 'organization/model_name'")

        # Initialize loaders and vectore storage by setting the path
        self.loaders = os.path.join(ROOT_DIR, BASE_DIR)

    @property
    def loaders(self) -> List[PyMuPDFLoader]:
        """Gets the list of document loaders.

        Returns:
            List[PyMuPDFLoader]: The list of document loaders.
        """
        return self._loaders

    @loaders.setter
    def loaders(self, path: str):
        """Sets up and initializes document loaders from the specified path.

        Args:
            path (str): The path to the directory containing document files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory '{path}' does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"'{path}' is not a directory.")

        # Initialize the loaders list
        self._loaders = []
        for file in os.listdir(path):
            if os.path.splitext(file)[-1].lower() in FILE_FORMATS:
                self._loaders.append(PyMuPDFLoader(os.path.join(path, file)))

    def risk_assessment(self):
        """Performs risk assessment logic for the agent.

        This method should be implemented to include specific logic for assessing risks
        based on the agent's configuration and loaded documents.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Risk assessment logic is not implemented yet.")
