import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI(title="RAG Application")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # will hold the FAISS index after upload


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == ".txt":
        loader = TextLoader(str(file_path))
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    store = FAISS.from_documents(chunks, embeddings)
    return store


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile" # or "llama3-8b-8192"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        docs = load_document(file_path)
        vector_store = build_vector_store(docs)
        
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(req: QueryRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a document first.")
    
    try:
        chain = get_qa_chain(vector_store)
        response = chain.invoke({"query": req.question})
        answer = response["result"]
        sources = [doc.page_content for doc in response["source_documents"]]
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
