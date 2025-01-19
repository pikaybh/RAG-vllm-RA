import os

from langchain_core.runnables import Runnable

from utils import timer
from templates import BaseDocuments, KrasRiskAssessmentOutput, kras_map


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
    manual_paths = self.get_docs_paths(BaseDocuments.name, BaseDocuments.manual)
    manual_docs = self.load_documents(manual_paths)
    manual_vectorstores = self.create_vectorstore(manual_docs)
    manual_retriever = self.create_retriever(manual_vectorstores, input_map=kras_map, search_type="similarity", search_kwargs={"k": 3})
    
    # Retriever Configuration for Searching Reference
    ref_paths = self.get_docs_paths(BaseDocuments.name, BaseDocuments.reference)
    ref_docs = self.load_documents(ref_paths)
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


__all__ = ["ra_chain"]