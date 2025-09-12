from typing import List
from langchain.schema import Document

class BaseRetriever:
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        raise NotImplementedError
