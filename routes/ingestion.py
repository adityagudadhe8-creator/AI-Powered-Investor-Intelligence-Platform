from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path
import os
from uuid import uuid4
from langchain_openai import AzureOpenAIEmbeddings
from vectorstore.azure_ai_search import AzureAISearchVectorStore
from ingestion.ingest_documents import ingest_document

router = APIRouter()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF documents are supported.")

    # Use a generated name so a client-supplied path cannot escape the upload directory.
    original_name = Path(file.filename or "report.pdf").name
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF documents are supported.")
    # Keep the stored name unique; ingestion receives the original report name below.
    stored_name = f"{uuid4().hex}_{original_name}"

    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / stored_name
    total_bytes = 0
    try:
        with file_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB upload limit.")
                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        with file_path.open("rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        # Initialize embeddings and vector store
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

        vector_store = AzureAISearchVectorStore(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        )

        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store,
            source_name=original_name,
        )
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail="Document processing failed. Check service configuration and try again.",
        ) from exc
    finally:
        await file.close()

    return {
        "message": "Document uploaded successfully",
        "file_name": original_name
    }
