import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAIEmbeddings

from vectorstore.azure_ai_search import AzureAISearchVectorStore, Retriever
from llm.azure_openai import get_openai_client

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    company: str | None = Field(default=None, min_length=1, max_length=200)
    year: int | None = Field(default=None, ge=1900, le=2100)

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize vector store and retriever
        vector_store = AzureAISearchVectorStore(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        )
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        retriever = Retriever(vector_store.client, embeddings)

        # Retrieve relevant context
        context = ""
        docs = retriever.invoke(
            query=request.question,
            company=request.company,
            year=request.year
        )
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build chat prompt – include retrieved context and the user question
        prompt = f"You are an expert financial analyst. Use the following context from corporate reports to answer the user's question. If the context does not contain relevant information, politely indicate that you do not have enough data.\n\nContext:\n{context}\n\nUser Question: {request.question}\n\nAnswer:"

        client = get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        # Keep provider credentials, endpoints, and internal diagnostics out of API responses.
        raise HTTPException(status_code=502, detail="The investor analysis service is temporarily unavailable.") from e
