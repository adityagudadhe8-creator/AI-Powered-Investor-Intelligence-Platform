import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAIEmbeddings

from llm.azure_openai import get_structured_completion
from vectorstore.azure_ai_search import AzureAISearchVectorStore, Retriever

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int | None
) -> str:
    """
    Retrieve broad financial context from the vector store.
    """
    query = (
        f"{company} fiscal year {year} annual report financial results: "
        "revenue, net income, operating income, cash flow from operating activities, "
        "total assets, total liabilities, principal risks, and growth drivers."
    )

    documents = retriever.invoke(
        query=query,
        company=company,
        year=year,
        top_k=20
    )
    if not documents:
        raise ValueError(
            f"No report passages were retrieved for {company} {year}. "
            "Check that the report chunks were indexed and that the search filters match."
        )
    context = "\n\n".join(
        doc.page_content
        for doc in documents
        if doc.page_content and doc.page_content.strip()
    )
    if not context:
        raise ValueError(
            f"Search returned no usable report text for {company} {year}. "
            "Check the indexed content field and PDF text extraction."
        )
    return context


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    return f"""
You are an expert financial analyst extracting values from the supplied annual report passages.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null if unavailable.
- Search across every supplied passage for the requested metric; values may appear in tables or prose and may use millions or billions.
- If a metric appears in the context, extract it rather than returning null. Preserve the report's currency and units.
- Return only the value for the requested fiscal year ({year}); do not return values for other years.
- Each financial metric must be a single string, number, or null. Never return an object, dictionary, or year-to-value mapping for a financial metric.
- Financial values must match the report exactly, including units and currency when stated.
- Risk factors should be concise.
- Growth drivers should be concise.
- Return valid JSON only.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using RAG.
    """
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )

    result = metrics.model_dump()
    if not any(value is not None for value in result.values()):
        raise ValueError(
            f"The model could not extract any financial metrics for {company} {year}. "
            "Review the retrieved passages and the model deployment before saving results."
        )
    return result


def main() -> None:
    company = "Apple"
    year = 2024

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    vector_store = AzureAISearchVectorStore(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        api_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
    )

    retriever = Retriever(vector_store.client, embeddings)

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)


    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__ == "__main__":
    main()
