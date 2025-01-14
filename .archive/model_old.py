'''
"""TODO: 고민 좀 해보기..."""

import os
from typing import List

from langchain_core.vectorstores import FAISS


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
        self._loaders = {}
        for file in os.listdir(path):
            if os.path.splitext(file)[-1].lower() in FILE_FORMATS:
                name, _ = os.path.splitext(file)
                loader = PyMuPDFLoader(os.path.join(path, file))
                self._loaders[name] = loader

    def risk_assessment(self):
        """Performs risk assessment logic for the agent.

        This method should be implemented to include specific logic for assessing risks
        based on the agent's configuration and loaded documents.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Risk assessment logic is not implemented yet.")
'''







'''
        # Build Vector DB
        self._build_ra_db()

        # Initialize Retriever
        self.retreiver = self.ra_db.as_retriever()
        self.docs = self.retreiver.get_relevant_documents(query)

    def _build_ra_db(self):
        data = self.loaders.values.load(),  # docucments
        print(data)

        documents = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
            encoding_name='cl100k_base'
        ).split_documents(data[0])

        self.ra_db = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

    def risk_assessment(self) -> object:
        prompt = ChatPromptTemplate.from_messages(
            list(
                build_prompt(
                    task="risk assessment",
                    steps="init"
                ).items()
            )
        )
        
        return prompt | self.model | self.retreiver



def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])
'''