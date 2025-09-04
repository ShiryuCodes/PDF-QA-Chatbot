import streamlit as st
import os
import tempfile
import asyncio
import nest_asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Fix for Streamlit async issues
try:
    nest_asyncio.apply()
except:
    pass

# Create event loop if none exists
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Page configuration
st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF QA Chatbot with Gemini")
st.markdown("Upload a PDF and ask questions about its content!")

# Initialize session state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def load_pdf_async(file_path):
    """Async function to load PDF"""
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def process_pdf(uploaded_file, api_key):
    """Process the uploaded PDF and create vector database"""
    try:
        # Set API key
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF using asyncio
        with st.spinner("Loading PDF..."):
            pages = asyncio.run(load_pdf_async(tmp_file_path))
        
        # Split documents
        with st.spinner("Splitting documents..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
        
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Create vector database
        with st.spinner("Building vector database..."):
            vectordb = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings_model,
                persist_directory=f"vector_db_{uploaded_file.name.replace('.pdf', '')}"
            )
            vectordb.persist()
        
        # Initialize chat model
        with st.spinner("Initializing chat model..."):
            model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Create QA chain
        prompt = hub.pull("rlm/rag-prompt")
        qa_chain = (
            {
                "context": vectordb.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectordb, qa_chain, len(chunks)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None, 0

# Sidebar for inputs
with st.sidebar:
    st.header("Setup")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key", 
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    # PDF upload
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf",
        help="Upload a PDF file to analyze"
    )
    
    # Process button
    if st.button("Process PDF", disabled=not (api_key and uploaded_file)):
        if api_key and uploaded_file:
            vectordb, qa_chain, num_chunks = process_pdf(uploaded_file, api_key)
            if vectordb and qa_chain:
                st.session_state.vectordb = vectordb
                st.session_state.qa_chain = qa_chain
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []  # Reset chat history
                st.success(f"✅ PDF processed successfully!")
                st.info(f"📄 Document split into {num_chunks} chunks")
            else:
                st.session_state.pdf_processed = False
    
    # Clear button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.pdf_processed and st.session_state.qa_chain:
    st.success("🟢 Ready to answer questions!")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.divider()
    
    # Question input
    question = st.text_input(
        "Ask a question about the PDF:",
        placeholder="What is this document about?",
        key="question_input"
    )
    
    # Answer question
    if question and st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.qa_chain.invoke(question)
                st.session_state.chat_history.append((question, answer))
                st.rerun()  # Refresh to show the new Q&A
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

elif not st.session_state.pdf_processed:
    st.info("👈 Please provide your Gemini API key and upload a PDF file to get started!")
    
    # Instructions
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Get a Gemini API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your free API key
        2. **Enter API Key**: Paste your API key in the sidebar
        3. **Upload PDF**: Choose a PDF file to analyze
        4. **Process**: Click "Process PDF" to prepare the document
        5. **Ask Questions**: Once processed, ask any questions about the PDF content
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ by Shivang",
    help="This app uses RAG (Retrieval Augmented Generation) to answer questions about your PDF documents"
)
