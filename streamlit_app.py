import streamlit as st
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Import core logic from main.py
from main import (
    load_document, 
    build_vector_store, 
    get_embeddings, 
    get_qa_chain, 
    INDEX_PATH, 
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    memory
)
from langchain_community.vectorstores import FAISS

# Page configuration
st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load existing index on startup if not in session state
if "vector_store" not in st.session_state:
    if Path(INDEX_PATH).exists():
        try:
            st.session_state.vector_store = FAISS.load_local(
                INDEX_PATH, 
                get_embeddings(), 
                allow_dangerous_deserialization=True
            )
            st.success("Loaded existing FAISS index from disk.")
        except Exception as e:
            st.error(f"Failed to load existing index: {e}")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# Sidebar for Document Management
with st.sidebar:
    st.title("📂 Document Manager")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                docs = load_document(file_path)
                new_store = build_vector_store(docs)
                
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = new_store
                else:
                    st.session_state.vector_store.merge_from(new_store)
                
                # Persist to disk
                st.session_state.vector_store.save_local(INDEX_PATH)
                st.success(f"Successfully processed {uploaded_file.name}")
                st.rerun() # Refresh to update list
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    
    # List Uploaded Documents
    st.subheader("My Documents")
    files = sorted([f.name for f in UPLOAD_DIR.iterdir() if f.is_file()])
    if files:
        for f in files:
            st.caption(f"📄 {f}")
    else:
        st.info("No documents uploaded yet.")
    
    st.divider()
    
    # Settings & Controls
    st.subheader("Controls")
    k_val = st.slider("Top-k Retrieval", min_value=1, max_value=10, value=3)
    if st.button("🗑️ Clear Conversation"):
        memory.clear()
        st.session_state.messages = []
        st.success("History cleared!")
        st.rerun()

# Main Chat Interface
st.title("🤖 AI RAG Assistant")
st.markdown("Ask anything about your uploaded documents.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is in the document?"):
    if st.session_state.vector_store is None:
        st.warning("Please upload a document first.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Setup the chain
            # We don't use the streaming callback here because Streamlit's 
            # st.write_stream is easier with a generator.
            # However, since get_qa_chain returns a chain, we'll use a standard generator.
            
            chain = get_qa_chain(st.session_state.vector_store, k=k_val)
            
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": prompt})
                answer = response["answer"]
                sources = response.get("source_documents", [])
            
            # Simulated streaming for better UX
            import time
            def stream_data():
                for word in answer.split(" "):
                    yield word + " "
                    time.sleep(0.05)
            
            st.write_stream(stream_data())
            
            # Display sources
            if sources:
                with st.expander("View Referenced Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**[{i+1}] {Path(doc.metadata.get('source', 'Unknown')).name}**")
                        st.caption(doc.page_content[:200] + "...")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
