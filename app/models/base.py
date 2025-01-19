import os
import inspect
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional
from abc import ABC, abstractmethod

from pydantic import Field
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Chroma

from templates import KrasRiskAssessmentOutput, kras_map
from utils import PromptBuilder, ext_map, get_logger, isext, mapper, timer


SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
DOC_DIR = "assets"
DB_DIR = "db"

# Logger Configuration
logger = get_logger("models.base", ROOT_DIR)

# UserWarning 경고 비활성화
warnings.filterwarnings("ignore", category=UserWarning)



class BaseLanguageModel(ABC):
    def __init__(self, model_id: str, logger: Optional[logging.Logger] = logger):
        """Initializes the Agent with a model identifier and loads documents.

        Args:
            logger (logging.Logger): The logger to use for logging.
            model_id (str): The model identifier in the format 'organization/model_name'.

        Raises:
            ValueError: If the model_id does not contain a '/' to split organization and model name.
        """
        self.logger = logger
        self.model_id = model_id

        try:
            self.organization, self.model_name = model_id.split("/")
        except ValueError:
            raise ValueError("model_id must be in the format 'organization/model_name'")

    def get_docs_paths(self, name: str, docs: Optional[List[str]] = None) -> List[str]:
        if name:
            return [os.path.join(ROOT_DIR, DOC_DIR, name, doc) for doc in docs]
        else:
            return [os.path.join(ROOT_DIR, DOC_DIR, doc) for doc in docs]

    def _get_loader(self, path: str, encoding: Optional[str] = "utf-8", *args, **kwargs) -> Callable:
        if isext(path, *ext_map["pdf"]):
            return PyMuPDFLoader(path, *args, **kwargs)
        elif isext(path, *ext_map["csv"]):
            try:
                return CSVLoader(path, encoding=encoding, *args, **kwargs)
            except UnicodeDecodeError as e:
                self.logger.error(f"Encoding error for file: {path}. Error: {e}")
                raise ValueError(f"Encoding error for file: {csv_path}. Error: {e}")
        else:
            raise ValueError(f"Unsupported file type: {path}")

    @timer
    def load_documents(self, path: str | List[str]) -> List[Document]:
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

        _paths = [os.path.join(ROOT_DIR, DOC_DIR, file) for file in os.listdir(path)]

        if not _paths:
            raise FileNotFoundError(f"The directory '{path}' does not contain any file.")

        documents = []
        for _path in _paths:
            loader = _get_loader(_path)
            documents.extend(loader.load())

        return documents

    @timer
    def create_vectorstore(self, documents, engine: str = "inmemory"):
        if engine == "inmemory":
            return InMemoryVectorStore.from_documents(documents, self.embeddings)
        elif engine == "faiss":
            _engine = FAISS
        elif engine == "chroma":
            _engine = Chroma
        else:
            raise ValueError(f"{engine} not found. Supported engines = ['faiss', 'chroma']")

        # Load & Saves locolly if the engine is either faiss or chroma.
        """TODO: 현재 아래 코드는 db 저장할 때 계속 오류나는데, 왜 그런지 알아보고 아래 코드 살려야 함."""
        __stack = inspect.stack()   # 호출 스택 정보를 가져옵니다.
        __caller = __stack[1]       # 스택에서 바로 위의 호출자 정보 (현재 메서드를 호출한 메서드)

        db_path = os.path.join(
            ROOT_DIR, 
            DB_DIR,
            self.organization,
            __caller.function[:-6], 
            engine
        )

        if not os.path.exists(db_path):
            Path(db_path).mkdir(parents=True, exist_ok=True)
            vectorstores = _engine.from_documents(documents, self.embeddings)
            vectorstores.save_local(db_path)
        else:
            vectorstores = _engine.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)

        return vectorstores

    @timer
    def create_retriever(self, vectorstores: VectorStore, input_map: Dict[str, str], *args, **kwargs) -> Callable:
        """Creates a logging-enabled retriever function.
        
        Args:
            vectorstores: Vector store to create retriever from
            *args: Additional arguments for retriever creation
            **kwargs: Additional keyword arguments for retriever creation
            
        Returns:
            Callable: Function that performs and logs document retrieval
        
        Raises:
            TypeError: If the query is not a string or dictionary
        """
        _retriever = vectorstores.as_retriever(*args, **kwargs)
        
        @self.logger.pinfo("{func_name} found {args[0]} documents")
        def retrieve(query: str|dict) -> List[Document]:
            if isinstance(query, str):
                return _retriever.get_relevant_documents(query)
            elif isinstance(query, dict):
                query: str = mapper(input_map, *query.keys())
                return _retriever.get_relevant_documents(query)
            else:
                raise TypeError(f"Invalid query type: {type(query)}")
            
        return RunnableLambda(retrieve)

    def create_formatter(self) -> Callable:
        @self.logger.pinfo("{func_name} formatted {args[0]} documents")
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        return RunnableLambda(format_docs)

    @timer
    def create_prompt(self, task: str, method: str) -> Callable:
        """Creates a logging-enabled prompt function.
        
        Args:
            task: Task type for prompt building
            method: Method name for prompt template
            
        Returns:
            Callable: Function that generates and logs the complete prompt
        """
        _prompt = ChatPromptTemplate.from_messages(PromptBuilder(task)[method])
        
        @self.logger.pinfo("{func_name} generated: {result}")
        def prompt(*args, **kwargs) -> str:
            return _prompt.invoke(*args, **kwargs)
            
        return RunnableLambda(prompt)

    @timer
    def create_structured_output(self, output: str) -> Callable:
        _structured_output = self.model.with_structured_output(output)

        @self.logger.pinfo("{func_name} structured output: {result}")
        def structured_output(*args, **kwargs) -> str:
            return _structured_output.invoke(*args, **kwargs)
        
        return RunnableLambda(structured_output)

    #########################################################################################################
    #
    #
    #               
    #   Chains
    #
    #
    #
    #
    #########################################################################################################
    def silent_ra_chain(self, method: str = "init") -> Runnable:

        # Retriever Configuration for Searching Manual 
        manual_path = os.path.join(ROOT_DIR, DOC_DIR, "2023년도 위험성평가 및 안전보건관리체계 우수사례집.pdf")
        manual_loader = PyMuPDFLoader(manual_path)
        manual_docs = manual_loader.load()
        manual_vectorstores = InMemoryVectorStore.from_documents(manual_docs, self.embeddings)
        manual_retriever = manual_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Retriever Configuration for Searching Reference
        ref_path = os.path.join(ROOT_DIR, DOC_DIR, "KRAS_찬영수정_250114.csv")
        ref_loader = CSVLoader(file_path=ref_path, encoding="utf-8")
        ref_docs = ref_loader.load()
        ref_vectorstores = InMemoryVectorStore.from_documents(ref_docs, self.embeddings)
        ref_retriever = ref_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 7})

        # Select Data
        def select_data(key: str) -> str:
            def _select_data(data):
                return data[key]
            return _select_data

        # Map Dictionary to String
        def dict2str(data) -> str:
            return mapper(kras_map, *data.keys())

        # Formatter Configuration
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt Configuration   
        # prompt = ChatPromptTemplate.from_messages(PromptBuilder("risk assessment")["init"])
        prompt = self.create_prompt("risk assessment", method)

        # Output Configuration
        structured_output = self.model.with_structured_output(KrasRiskAssessmentOutput)

        # Chain Configuration
        chain = (
            RunnableParallel(
                {
                    "work_type": select_data("work_type"),      # 공종
                    "procedure": select_data("procedure"),      # 공정
                    "manual": dict2str | RunnablePassthrough() | manual_retriever | format_docs,     # 매뉴얼
                    "reference": dict2str | RunnablePassthrough() | ref_retriever | format_docs,     # 유사 작업
                }
            ) 
            | prompt 
            | structured_output
        )

        return chain


__all__ = ["BaseLanguageModel"]
