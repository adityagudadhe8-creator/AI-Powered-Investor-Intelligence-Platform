import uuid

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from types import SimpleNamespace


class AzureAISearchVectorStore:
    """Azure AI Search vector store."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str
    ) -> None:
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to Azure AI Search.
        """
        documents = []

        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)

            documents.append(
                {
                    "id": str(uuid.uuid4()),
                    "company": company,
                    "year": year,
                    "source_file": source_file,
                    "content": chunk.page_content,
                    "content_vector": vector
                }
            )

        result = self.client.upload_documents(documents)

        uploaded = sum(item.succeeded for item in result)

        print(f"Uploaded {uploaded}/{len(documents)} chunks.")

class Retriever:
    """Simple wrapper around Azure Search client for retrieving relevant chunks.
    Mirrors the Retriever used in the RAG extractor.
    """
    def __init__(self, client, embeddings=None):
        self.client = client
        self.embeddings = embeddings

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Retrieve relevant chunks from Azure AI Search.
        Returns a list of SimpleNamespace objects with `page_content`.
        """
        filter_expr = None
        filters = []
        if company:
            # Escape apostrophes per OData string-literal rules.
            escaped_company = company.replace("'", "''")
            filters.append(f"company eq '{escaped_company}'")
        if year is not None:
            filters.append(f"year eq '{year}'")
        if filters:
            filter_expr = " and ".join(filters)
        search_options = {
            "search_text": query,
            "top": top_k,
            "filter": filter_expr,
        }
        if self.embeddings is not None:
            search_options["vector_queries"] = [
                VectorizedQuery(
                    vector=self.embeddings.embed_query(query),
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ]

        results = self.client.search(**search_options)
        documents = []
        for result in results:
            content = result.get("content", "")
            documents.append(SimpleNamespace(page_content=content))
        return documents