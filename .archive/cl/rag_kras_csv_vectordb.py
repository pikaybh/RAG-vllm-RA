#%% 
#�ʿ��� ���̺귯��
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
import os
#%%
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
#%%
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
class CSVRowTextSplitter(TextSplitter):
    """
    Custom TextSplitter to split CSV rows into Documents.
    Each row is treated as a single chunk.
    """

    def __init__(self, include_headers=True):
        """
        Args:
            include_headers (bool): Whether to include CSV column headers in each row's content.
        """
        super().__init__()
        self.include_headers = include_headers

    def split_text(self, text):
        """
        Since this splitter works at row level, the input text will already represent a row.
        Hence, it simply returns the text as a single chunk.
        """
        return [text]

    def split_csv(self, file_path, encoding="utf-8"):
        """
        Reads a CSV file and splits it row by row into Document objects.

        Args:
            file_path (str): Path to the CSV file.
            encoding (str): Encoding of the CSV file.
        
        Returns:
            List[Document]: List of Document objects, one for each row.
        """
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding)
        
        # Optionally include headers
        headers = df.columns if self.include_headers else None

        # Convert each row into a Document
        documents = []
        for _, row in df.iterrows():
            if headers is not None:
                row_content = ' '.join([f"{col}: {row[col]}" for col in headers])
            else:
                row_content = ' '.join(map(str, row.values))
            documents.append(Document(page_content=row_content))
        
        return documents

    

#%%
################################################################
#csv ������ �ε�
csv_file_path = "./KRAS_��������_250114.csv"
loader = CSVLoader(file_path=csv_file_path)
# ������ �ε�
docs = loader.load()
#print(docs[20])

#%%
#�ؽ�Ʈ ����
# text_splitter = CSVRowTextSplitter(include_headers=True)
# split_docs = text_splitter.split_documents(docs)

# print(split_docs[10])

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")  # OpenAI �� ���
vectorstore = FAISS.from_documents(docs, embedding_model)

vectorstore.save_local("faiss_vectorstore")

# �˻� ����
query = "������з�(����): �����۾�"
results = vectorstore.similarity_search(query, k=7)

# �˻� ��� ���
for result in results:
    print(result.page_content)
# %%
