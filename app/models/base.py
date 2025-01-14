import os
import inspect
import warnings
from pathlib import Path
from typing import List, Literal
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import TransformChain

from utils import PromptBuilder, payloader, isext


SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
FILE_DIR = "assets"
DB_DIR = "db"


# UserWarning 경고 비활성화
warnings.filterwarnings("ignore", category=UserWarning)



class RiskItem(BaseModel):
    번호: int = Field(description="시리얼 숫자")
    위험요소: str = Field(description="식별된 위험 요소")
    위험성평가: str = Field(description="해당 위험 요소의 위험성 평가 결과")
    위험도: Literal["높음", "중간", "낮음"] = Field(description="해당 위험 요소의 위험도")
    감소대책: List[str] = Field(description="위험 요소 감소를 위해 권장되는 통제 및 제한 조치 목록")



class RiskAssessmentOutput(BaseModel):
    작업: str = Field(description="작업의 이름")
    위험성평가표: List[RiskItem] = Field(description="각 위험 요소에 대한 위험성 평가와 통제 조치 목록")
    기타: List[str] = Field(description="기타 제언")



class BaseLanguageModel(ABC):
    def __init__(self, model_id: str):
        """Initializes the Agent with a model identifier and loads documents.

        Args:
            model_id (str): The model identifier in the format 'organization/model_name'.

        Raises:
            ValueError: If the model_id does not contain a '/' to split organization and model name.
        """
        self.model_id = model_id

        try:
            self.organization, self.model_name = model_id.split("/")
        except ValueError:
            raise ValueError("model_id must be in the format 'organization/model_name'")

    def load_pdfs(self, path: str):
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

        pdf_paths = [os.path.join(ROOT_DIR, FILE_DIR, pdf_path) for pdf_path in os.listdir(path) if isext(pdf_path, "pdf")]

        if not pdf_paths:
            raise FileNotFoundError(f"The directory '{path}' does not contain any PDF file.")

        documents = []
        for pdf_path in pdf_paths:
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())

        return documents

    def create_vectorstore(self, documents, engine: str = "faiss"):
        if engine == "faiss":
            _engine = FAISS
        elif engine == "chroma":
            _engine = Chroma
        else:
            raise ValueError(f"{engine} not found. Supported engines = ['faiss', 'chroma']")
        
        return _engine.from_documents(documents, self.embeddings)

        """
        __stack = inspect.stack() # 호출 스택 정보를 가져옵니다.
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
        """

    def ra_chain(self, method: str = "init"):
        docucments = self.load_pdfs(os.path.join(ROOT_DIR, FILE_DIR))
        vectorstores = self.create_vectorstore(docucments)
        retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # 검색 체인 정의
        def search_transform(inputs):
            # 검색 결과를 context로 변환
            relevant_docs = retriever.invoke(inputs["input"])
            
            return {"context": "\n".join(doc.page_content for doc in relevant_docs)}

        search_chain = TransformChain(
            input_variables=["input"],  # 입력 변수
            output_variables=["context"],  # 출력 변수
            transform=search_transform  # 변환 함수
        )

        prompt = ChatPromptTemplate.from_messages(
            PromptBuilder("risk assessment")[method]
        )

        structured_output = self.model.with_structured_output(RiskAssessmentOutput)

        return search_chain | prompt | structured_output
        
        """
        # 체인을 인증 로직과 함께 래핑
        final_chain = search_chain | prompt | structured_output
        return SecureChainWrapper(final_chain)
        """
        
