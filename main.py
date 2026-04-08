import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import logging
import asyncio
from datetime import datetime

load_dotenv()

# --------------- Upgrade 12: Structured Logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application")

# --------------- State ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_PATH = "faiss_index"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

vector_store = None  # will hold the FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --------------- Upgrade 6: Conversation Memory ---------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# --------------- Upgrade 9: Query Caching ---------------
query_cache: dict[str, dict] = {}

# --------------- Upgrade 2: Persist the FAISS Index (Loading) ---------------
@app.on_event("startup")
async def startup_event():
    global vector_store
    if Path(INDEX_PATH).exists():
        try:
            vector_store = FAISS.load_local(
                INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded existing FAISS index from disk.")
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")


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
    """Split documents into chunks and build a FAISS vector store using Cosine Similarity."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        logger.warning("No text chunks were generated from the documents.")
        return None

    # Use Cosine Similarity to ensure scores are between 0 and 1 for thresholding
    store = FAISS.from_documents(
        chunks, 
        embeddings,
        distance_strategy="cosine"
    )
    return store


def get_qa_chain(store, k=3, streaming=False, callbacks=None):
    """Create a ConversationalRetrievalChain from the vector store."""
    # This LLM is used for internal rephrasing (condensing the question)
    # It should NOT use callbacks or streaming to avoid leaking internal steps to the UI.
    condense_llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile"
    )

    # This LLM is used for the final answer generation
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        streaming=streaming,
        callbacks=callbacks
    )
    
    # --------------- Upgrade 1: Prompt Templating ---------------
    template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # --------------- Upgrade 4 & 11: Configurable Retrieval & Score Threshold ---------------
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": 0.2
        }
    )
    
    # --------------- Upgrade 6: Conversation Memory ---------------
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# --------------- Upgrade 10: /documents Endpoint ---------------
@app.get("/documents")
async def list_documents():
    """List all uploaded files in the uploads directory."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            stats = f.stat()
            files.append({
                "name": f.name,
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
    return sorted(files, key=lambda x: x["name"])


class QueryRequest(BaseModel):
    question: str
    k: int = 3


@app.delete("/history")
async def clear_history():
    """Reset the conversation memory."""
    memory.clear()
    return {"message": "Conversation history cleared."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    
    # --------------- Upgrade 8: Input Validation & File-type Guard ---------------
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected upload of unsupported file type: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        logger.info(f"Receiving file upload: {file.filename}")
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        docs = load_document(file_path)
        if not docs or all(not doc.page_content.strip() for doc in docs):
            logger.warning(f"Document {file.filename} appears to be empty or image-only.")
            raise HTTPException(
                status_code=400, 
                detail=f"Successfully uploaded {file.filename}, but no text could be extracted. It might be an image-only PDF or empty."
            )

        new_store = build_vector_store(docs)
        if new_store is None:
             raise HTTPException(status_code=400, detail="Could not generate search chunks from this document.")

        # --------------- Upgrade 5: Multi-document Support ---------------
        if vector_store is None:
            vector_store = new_store
        else:
            vector_store.merge_from(new_store)
            logger.info(f"Merged {file.filename} into existing vector store.")

        # --------------- Upgrade 2: Persist the FAISS Index (Saving) ---------------
        vector_store.save_local(INDEX_PATH)
        logger.info(f"Vector store saved to {INDEX_PATH}")
        
        return {"message": f"Successfully processed and merged {file.filename}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing upload {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


from langchain.callbacks import AsyncIteratorCallbackHandler

@app.post("/query", response_class=StreamingResponse, responses={200: {"content": {"text/plain": {}}}})
async def query_document(req: QueryRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a document first.")
    
    # --------------- Upgrade 9: Query Caching ---------------
    cache_key = f"{req.question.strip().lower()}_k{req.k}"
    if cache_key in query_cache:
        logger.info(f"Cache hit for question: {req.question}")
        async def cached_response():
            yield query_cache[cache_key]
        return StreamingResponse(cached_response(), media_type="text/plain")

    async def stream_generator(question: str, k: int):
        callback = AsyncIteratorCallbackHandler()
        chain = get_qa_chain(vector_store, k=k, streaming=True, callbacks=[callback])
        
        # Start the chain in a background task
        task = asyncio.create_task(chain.ainvoke({"question": question}))
        
        full_response = ""
        async for token in callback.aiter():
            full_response += token
            yield token
            
        # Ensure the task is completed and get response (for sources)
        response = await task
        
        # Append sources at the end
        sources_text = ""
        if "source_documents" in response:
            sources_text += "\n\nSources:\n"
            for i, doc in enumerate(response["source_documents"]):
                name = doc.metadata.get("source", "Unknown")
                sources_text += f"[{i+1}] {Path(name).name}\n"
        
        yield sources_text
        
        # Cache the result
        query_cache[cache_key] = full_response + sources_text

    try:
        logger.info(f"Processing streaming query: {req.question} with k={req.k}")
        return StreamingResponse(
            stream_generator(req.question, req.k), 
            media_type="text/plain"
        )
    except Exception as e:
        logger.exception(f"Error querying: {req.question}")
        raise HTTPException(status_code=500, detail=str(e))
