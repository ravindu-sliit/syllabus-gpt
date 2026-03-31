import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _collect_pdfs(target_pdf: str | None) -> list[Path]:
    if target_pdf:
        pdf_path = Path(target_pdf)
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF not found or invalid: {pdf_path}")
        return [pdf_path]

    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("data directory not found")

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF files found in data/")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF files into a FAISS index")
    parser.add_argument("--pdf", help="Optional path to a specific PDF to ingest")
    args = parser.parse_args()

    load_dotenv()
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to .env before ingesting.")

    pdf_files = _collect_pdfs(args.pdf)
    print(f"Found {len(pdf_files)} PDF file(s) to ingest")

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    print("Saved vector index to faiss_index/")


if __name__ == "__main__":
    main()