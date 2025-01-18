import os
import inspect
import warnings
from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import Field
from langchain.chains import TransformChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableParallel, chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Chroma

from utils import PromptBuilder, payloader, isext, get_logger, format_docs
from templates import RiskAssessmentOutput, KrasRiskAssessmentOutput
from utils.decorators import timer


SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
FILE_DIR = "assets"
DB_DIR = "db"

# Logger Configuration
logger = get_logger("models.base", ROOT_DIR)

# UserWarning 경고 비활성화
warnings.filterwarnings("ignore", category=UserWarning)



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

    # TODO: load_ 계열 Method들 통합하여 추상화하기기
    @timer
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
                raise

            # loader = CSVLoader(csv_path)
            # documents.extend(loader.load())

        return documents

    @timer
    def create_vectorstore(self, documents, engine: str = "inmemory"):
        if engine == "inmemory":
            _engine = InMemoryVectorStore
        elif engine == "faiss":
            _engine = FAISS
        elif engine == "chroma":
            _engine = Chroma
        else:
            raise ValueError(f"{engine} not found. Supported engines = ['faiss', 'chroma']")
        
        return _engine.from_documents(documents, self.embeddings)

        """TODO: db 저장할 때 계속 오류나는데 왜 그런지 알아보고 아래 코드 살리기
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
        

    def kras_chain(self, method: str = "kras"):
        docucments = self.load_csvs(os.path.join(ROOT_DIR, FILE_DIR))
        vectorstores = self.create_vectorstore(docucments)
        retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 7})
        
        # 검색 체인 정의
        def search_transform(inputs):
            # 검색 결과를 context로 변환
            _input = inputs["input"]
            relevant_docs = retriever.invoke(_input)
            context = "\n".join(doc.page_content for doc in relevant_docs)
            logger.info(f"Search output (context): {context}")
            return {"context": context, "input": _input}

        search_chain = TransformChain(
            input_variables=["input"],  # 입력 변수
            output_variables=["context"],  # 출력 변수
            transform=search_transform  # 변환 함수
        )

        template = PromptBuilder("risk assessment")[method]
        logger.info(f"Template: {template}")

        prompt = ChatPromptTemplate.from_messages(
            template
        )

        structured_output = self.model  # .with_structured_output(KrasRiskAssessmentOutput)
        logger.info(f"Structured output: {structured_output}")
        
        return search_chain | prompt | structured_output

    def kras_chain2(self, method: str = "kras"):
        # from langchain_core.output_parsers import StrOutputParser
        docucments = self.load_csvs(os.path.join(ROOT_DIR, FILE_DIR))
        vectorstores = self.create_vectorstore(docucments)
        retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 7})

        template = PromptBuilder("risk assessment")[method]
        logger.info(f"Template: {template}")
        
        prompt = ChatPromptTemplate.from_messages(template)
        logger.info(f"{prompt = }")

        structured_output = self.model.with_structured_output(KrasRiskAssessmentOutput)
        logger.info(f"Structured output: {structured_output}")
        
        chain = (
            {"context": retriever | format_docs, "topic": RunnablePassthrough()}  # 문맥 검색기
            | prompt
            | structured_output
            # | StrOutputParser()
        )

        logger.info(f"{chain = }")

        return chain

    @timer
    def full_chain(self, method: str = "full") -> Runnable:
        # Search Manual 
        logger.info("Loading PDF documents for manual...")
        manual_docs = self.load_pdfs(os.path.join(ROOT_DIR, FILE_DIR))
        manual_vectorstores = self.create_vectorstore(manual_docs)
        manual_retriever = manual_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logger.info("Manual retriever setup complete")
        
        # Search Reference
        logger.info("Loading CSV documents for reference...")
        ref_docs = self.load_csvs(os.path.join(ROOT_DIR, FILE_DIR))
        ref_vectorstores = self.create_vectorstore(ref_docs)
        ref_retriever = ref_vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 7})
        logger.info("Reference retriever setup complete")

        # Prompt Configuration   
        logger.info("Setting up prompt template...")
        prompt = ChatPromptTemplate.from_messages(PromptBuilder("risk assessment")[method])
        logger.info(f"Prompt template: {prompt}")

        # Output Configuration
        logger.info("Configuring structured output...")
        structured_output = self.model.with_structured_output(KrasRiskAssessmentOutput)

        # Chain Configuration
        logger.info("Building chain...")
        chain = RunnableParallel(
            {
                "work_type": lambda x: x["work_type"],
                "procedure": lambda x: x["procedure"],
                # "manual": lambda x: format_docs(manual_retriever.get_relevant_documents(x["work_type"] + " " + x["procedure"])),
                # "reference": lambda x: format_docs(ref_retriever.get_relevant_documents(x["work_type"] + " " + x["procedure"]))
                "manual": lambda x: (
                    logger.info(f"Searching manual for: {x['work_type']} {x['procedure']}"),
                    format_docs(manual_retriever.get_relevant_documents("공종: " + x["work_type"] + " 공정: " + x["procedure"]))
                )[1],
                "reference": lambda x: (
                    logger.info(f"Searching reference for: {x['work_type']} {x['procedure']}"),
                    format_docs(ref_retriever.get_relevant_documents("공종: " + x["work_type"] + " 공정: " + x["procedure"]))
                )[1]
            }
        ) | prompt | structured_output
        logger.info("Chain setup complete")

        return chain

__all__ = ["BaseLanguageModel"]
