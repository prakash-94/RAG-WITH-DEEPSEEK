# PHASE 2: CREATE A VECTOR DATABASE (MEMORY OF LLM)
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#STEP 1: UPLOAD AND LOAD RAW PDF FILE

pdfs_directory = 'pdfs'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = "universal_declaration_of_human_rights.pdf"
documents = load_pdf(file_path)
print("PDF pages: ",len(documents))


#STEP 2: CREATE CHUNKS
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    ) 
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
print("Chunk_counts: ", len(text_chunks))

#STEP 3: GENERATE EMBEDDING MODELS (USING DEEPSEEK RI WITH OLLAMA)
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model = ollama_model_name)
    return embeddings
#STEP 4: INDEX DOCUMENTS **STORE EMBEDDINGS IN FAISS(VECTOR BASE)
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)