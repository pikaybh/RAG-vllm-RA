import os
import base64
import inspect
import logging
import pathlib
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Required
from abc import ABC, abstractmethod

from pydantic import Field
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
# from langchain_community.chains import chain
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, JSONLoader, UnstructuredXMLLoader, TextLoader
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

    def get_docs_paths(self, name: Optional[str] = None, docs: Required[List[str]] = None) -> List[str]:
        if name:
            return [os.path.join(ROOT_DIR, DOC_DIR, name, doc) for doc in docs]
        else:
            return [os.path.join(ROOT_DIR, DOC_DIR, doc) for doc in docs]

    def _get_loader(self, path: str, 
            # CSVLoader
            encoding: Optional[str] = "utf-8", 
            autodetect_encoding: bool = True,
            # JSONLoader
            jq_schema='.', 
            text_content=False, 

            # UnstructuredXMLLoader
            mode="single",
            strategy="fast",
            
            *args, **kwargs
        ) -> Callable:

        if isext(path, "pdf"):  # if isext(path, *ext_map["pdf"]):
            return PyMuPDFLoader(path, *args, **kwargs)

        elif isext(path, "csv"):
            try:
                return CSVLoader(path, encoding=encoding, *args, **kwargs)
            except UnicodeDecodeError as e:
                self.logger.error(f"Encoding error for file: {path}. Error: {e}")
                raise ValueError(f"Encoding error for file: {csv_path}. Error: {e}")
        elif isext(path, "json"):
            return JSONLoader(path, jq_schema=jq_schema, text_content=text_content, *args, **kwargs)  # JSON 문서에서 'content' 필드를 추출
        elif isext(path, "txt"):
            return TextLoader(path, encoding=encoding, autodetect_encoding=autodetect_encoding, *args, **kwargs)
        elif isext(path, "xml"):
            return UnstructuredXMLLoader(path, mode=mode, strategy=strategy, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    @timer
    def load_documents(self, paths: str | List[str], split_delimiter: Optional[str] = None) -> List[Document]:
        """Sets up and initializes document loaders from the specified path.

        Args:
            path (str): The path to the directory containing document files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if isinstance(paths, str):
            _tmp = paths
            paths = []
            paths.append(_tmp)

        if not paths[0]:
            raise ValueError("paths is required")

        if not os.path.exists(paths[0]):
            raise FileNotFoundError(f"The directory '{paths[0]}' does not exist.")

        _paths = [os.path.join(ROOT_DIR, DOC_DIR, file) for file in paths]

        documents = []
        for _path in _paths:
            loader = self._get_loader(_path)
            documents.extend(loader.load())

        # Split documents by delimiter
        if split_delimiter:
            _documents = []
            for doc in documents:
                _documents.extend(doc.page_content.split(split_delimiter))
            documents = _documents

        return documents

    
    @timer
    def load_vectorstore(self, db_path: str, engine: str = "faiss"):
        _path = str(pathlib.Path(DB_DIR, db_path))

        self.logger.info(f"{_path = }")

        if engine == "faiss":
            return FAISS.load_local(_path, self.embeddings, allow_dangerous_deserialization=True)
        elif engine == "chroma":
            return Chroma.load_local(_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            raise ValueError(f"{engine} not found. Supported engines = ['faiss', 'chroma']")

    @timer
    def create_vectorstore(self, documents, engine: str = "inmemory"):
        """Creates a vector store from documents.
    
        Args:
            documents: List of documents or strings
            
        Returns:
            VectorStore: Vector store containing the documents
        """
        # 문서를 적절한 크기로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,          # 각 청크의 최대 길이
            chunk_overlap=200,        # 청크 간 중복되는 문자 수
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # 분할 우선순위
            is_separator_regex=False
        )
        
        
        # 문자열 리스트인 경우 Document 객체로 변환
        if documents and isinstance(documents[0], str):
            documents = [Document(page_content=text) for text in documents]
    
            # 문서 분할
            documents = text_splitter.split_documents(documents)

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

        db_path = str(pathlib.Path(
            # ROOT_DIR, 
            DB_DIR,
            self.organization,
            __caller.function[:-6], 
            engine
        ))

        self.logger.info(f"{db_path = }")

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

    def encode_image(self, image_path: str) -> Callable:
        """
        Encode image to base64

        Args:
            image_path (str): The path to the image file.
            
        Returns:
            str: The base64 encoded image string.
        """
        with open(image_path, "rb") as image_file:
            img_bytes = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:image/jpeg;base64,{img_bytes}"

    '''
    @chain
    def vision_chain(self, inputs):
        """
        TODO: 비전 input 구현할 것.
            지금은 아프고 귀찮아서 
            못하겠어요. 히읗
            ref: https://teddylee777.github.io/langchain/langchain-code-generator/
        """
        image_path, url = str(inputs["image_path"]), bool(inputs["url"])
        if url:
            image_url = image_path
        else:
            base64_image = self.encode_image(image_path)
            image_url = f"{base64_image}"
        system_message = SystemMessage(
            content=[
                """Write some python code to solve the user's problem. 

                Return only python code in Markdown format, e.g.:

                ```python
                ....
                ```"""
            ]
        )
        vision_message = HumanMessage(
            content=[
                {"type": "text", "text": "Can you write a python code to draw this plot?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "auto",
                    },
                },
            ]
        )
        output = self.model.invoke([system_message, vision_message])

        return output.content
    '''

    @timer
    def create_structured_output(self, output: str) -> Callable:
        _structured_output = self.model.with_structured_output(output)

        @self.logger.pinfo("{func_name} structured output: {result}")
        def structured_output(*args, **kwargs) -> str:
            return _structured_output.invoke(*args, **kwargs)
        
        return RunnableLambda(structured_output)

    #########################################################################################################
    #                                                                                                       #
    #                                                                                                       #
    #                                                                                                       #
    #                                                                                                       #
    #                                              Chains                                                   #
    #                                                                                                       #
    #                                                                                                       #
    #                                                                                                       #
    #                                                                                                       #
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
