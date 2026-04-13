from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PDFS_DIRECTORY = BASE_DIR / "pdfs"
VECTORSTORE_DIRECTORY = BASE_DIR / "vectorstore"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"


def save_uploaded_pdf(uploaded_file):
    PDFS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = PDFS_DIRECTORY / uploaded_file.name

    with destination.open("wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())

    return destination


def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(str(file_path))
    return loader.load()


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def get_embedding_model(model_name=OLLAMA_MODEL_NAME):
    return OllamaEmbeddings(model=model_name)


def _vectorstore_path_for(pdf_path):
    return VECTORSTORE_DIRECTORY / pdf_path.stem


def build_vector_store(pdf_path):
    documents = load_pdf_documents(pdf_path)
    text_chunks = create_chunks(documents)
    vectorstore_path = _vectorstore_path_for(pdf_path)
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(str(vectorstore_path))
    return faiss_db


def load_vector_store(pdf_path):
    vectorstore_path = _vectorstore_path_for(pdf_path)
    index_file = vectorstore_path / "index.faiss"
    store_file = vectorstore_path / "index.pkl"

    if not index_file.exists() or not store_file.exists():
        return None

    return FAISS.load_local(
        str(vectorstore_path),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def get_or_create_vector_store(uploaded_file):
    pdf_path = save_uploaded_pdf(uploaded_file)
    existing_store = load_vector_store(pdf_path)

    if existing_store is not None:
        return pdf_path, existing_store

    return pdf_path, build_vector_store(pdf_path)
