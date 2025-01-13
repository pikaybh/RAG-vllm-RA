import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.base import BaseModel
from utils import build_prompt, payloader


load_dotenv()



class OpenAIModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)

        # Initialize Engine
        self.model = ChatOpenAI(
            model=self.model_name, 
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize Embedding Model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Build Vector DB
        self._build_ra_db()

        # Initialize Retriever
        self.retreiver = self.ra_db.as_retriever()
        self.docs = self.retreiver.get_relevant_documents(query)

    def _build_ra_db(self):
        data = self.loaders[0].load(),  # docucments
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

if __name__ == "__main__":
    model = OpenAIModel(model_id="openai/gpt-4o")
    gpt_rag_chain = model.risk_assessment()
    gpt_rag_chain.invoke({"": (format_docs()), "input": "철근 배근 작업"})

