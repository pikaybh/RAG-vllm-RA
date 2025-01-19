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
FILE_DIR = "assets"
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

        _paths = [os.path.join(ROOT_DIR, FILE_DIR, file) for file in os.listdir(path)]

        if not _paths:
            raise FileNotFoundError(f"The directory '{path}' does not contain any file.")

        documents = []
        for _path in _paths:
            loader = _get_loader(_path)
            documents.extend(loader.load())

        return documents

    # TODO: load_ 계열 Method들 통합하여 추상화하기기
    @timer
    def load_csvs(self, path: str):
        """Sets up and initializes document loaders for CSV files from the specified path.

        Args:
            path (str): The path to the directory containing CSV files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory '{path}' does not exist.")

        if not os.path.isdir(path):
            raise NotADirectoryError(f"'{path}' is not a directory.")

        csv_paths = [os.path.join(path, csv_file) for csv_file in os.listdir(path) if csv_file.endswith('.csv')]

        if not csv_paths:
            raise FileNotFoundError(f"The directory '{path}' does not contain any CSV file.")

        documents = []
        for csv_path in csv_paths:
            try:
                # Specify encoding explicitly
                loader = CSVLoader(file_path=csv_path, encoding="utf-8")
                documents.extend(loader.load())
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error for file: {csv_path}. Error: {e}")
                raise ValueError(f"Encoding error for file: {csv_path}. Error: {e}")

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

    @timer
    def ra_chain(self, method: str = "init") -> Runnable:
        """Configures the RAG risk assessment processing chain.
        
        Creates a processing chain that:
        1. Routes the input type using type_router
        2. Processes work information and retrieves relevant documents in parallel
        3. Formats the results using the prompt template
        4. Structures the output into the desired format
        
        The chain expects input in the format:
            ```json
            {
                "work_type": str,
                "procedure": str
            }
            ```
        
        The chain expects input in the format on server:
            ```python
            class KrasRiskAssessmentInput(BaseModel):
                work_type: str = Field(description="작업 공종의 이름")
                procedure: str = Field(description="작업 공정의 이름")
            ```
        
        The chain returns the following output:
            ```python
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
            ``` 
        
        Returns:
            A chain that processes construction work information and returns
            structured risk assessment results.
        """

        # Retriever Configuration for Searching Manual 
        manual_docs = self.load_pdfs(os.path.join(ROOT_DIR, FILE_DIR))
        manual_vectorstores = self.create_vectorstore(manual_docs)
        manual_retriever = self.create_retriever(manual_vectorstores, input_map=kras_map, search_type="similarity", search_kwargs={"k": 3})
        
        # Retriever Configuration for Searching Reference
        ref_docs = self.load_csvs(os.path.join(ROOT_DIR, FILE_DIR))
        ref_vectorstores = self.create_vectorstore(ref_docs)
        ref_retriever = self.create_retriever(ref_vectorstores, input_map=kras_map, search_type="similarity", search_kwargs={"k": 7})

        # Formatter Configuration
        formatter = self.create_formatter()

        # Prompt Configuration   
        prompt = self.create_prompt("risk assessment", method)

        # Output Configuration
        structured_output = self.create_structured_output(KrasRiskAssessmentOutput)

        # Chain Configuration
        chain = (
            RunnableParallel(
                {
                    "work_type": lambda x: x["work_type"],      # 공종
                    "procedure": lambda x: x["procedure"],      # 공정
                    "manual": manual_retriever | formatter,     # 매뉴얼
                    "reference": ref_retriever | formatter,     # 유사 작업
                }
            ) 
            | prompt 
            | structured_output
        )

        return chain

    def silent_ra_chain(self, method: str = "init") -> Runnable:

        # Retriever Configuration for Searching Manual 
        manual_path = os.path.join(ROOT_DIR, FILE_DIR, "2023년도 위험성평가 및 안전보건관리체계 우수사례집.pdf")
        manual_loader = PyMuPDFLoader(manual_path)
        manual_docs = manual_loader.load()
        manual_vectorstores = InMemoryVectorStore.from_documents(manual_docs, self.embeddings)
        manual_retriever = manual_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Retriever Configuration for Searching Reference
        ref_path = os.path.join(ROOT_DIR, FILE_DIR, "KRAS_찬영수정_250114.csv")
        ref_loader = CSVLoader(file_path=ref_path, encoding="utf-8")
        ref_docs = ref_loader.load()
        ref_vectorstores = InMemoryVectorStore.from_documents(ref_docs, self.embeddings)
        ref_retriever = ref_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 7})

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
                    "work_type": lambda x: x["work_type"],      # 공종
                    "procedure": lambda x: x["procedure"],      # 공정
                    "manual": dict2str | RunnablePassthrough() | manual_retriever | format_docs,     # 매뉴얼
                    "reference": dict2str | RunnablePassthrough() | ref_retriever | format_docs,     # 유사 작업
                }
            ) 
            | prompt 
            | structured_output
        )

        return chain

__all__ = ["BaseLanguageModel"]
