import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit App Title and Branding
st.set_page_config(page_title="SmartDocuBot 🤖📄", page_icon="📄")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #2E86AB;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #eef1f5;
        }
        .stButton > button {
            background-color: #2E86AB;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
        .stTextInput>div>div>input {
            padding: 0.75rem;
            border-radius: 8px;
        }
        .chat-bubble {
            background: #e1ecf4;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .chat-title {
            font-weight: bold;
            color: #2E86AB;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">SmartDocumentBot</div>', unsafe_allow_html=True)
st.markdown("### I'm your personal chatbot assistant. Upload a document, and I'll answer your questions based on its content.")

# Sidebar Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])
    proceed = st.button("Proceed")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if uploaded_file and proceed:
    try:
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Loading and processing document..."):
            # Load the document
            loader = PyPDFLoader("temp_doc.pdf")
            documents = loader.load()

            # Split the document
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Embed the chunks
            embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.from_documents(chunks, embedder)

            # Create retriever and LLM
            retriever = vectordb.as_retriever()
            llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

            # Define the prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful assistant that only answers questions based on the given context.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
            )

            # Create retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=False
            )
            
            st.session_state.qa_chain = qa_chain
            st.success("Document processed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
    
# Chat UI
if st.session_state.qa_chain:
    st.markdown("### Ask anything about your document")
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("Type your question:", value=st.session_state.input_text, key="query_input")
    with col2:
        generate = st.button("Generate")

    if generate and user_query.strip():
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.qa_chain({"query": user_query})
                answer = response["result"]
                st.session_state.chat_history.append((user_query, answer))
                st.session_state.input_text = ""  # Clear query
                st.rerun()  # Reset input field visually
            except Exception as e:
                st.error(f" Error while answering: {e}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("###  Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="chat-bubble"><div class="chat-title">You:</div> {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble"><div class="chat-title">SmartDocuBot:</div> {a}</div>', unsafe_allow_html=True)

    # Download history
    chat_log = "\n\n".join([f"You: {q}\nSmartDocuBot: {a}" for q, a in st.session_state.chat_history])
    st.download_button(" Download Chat", chat_log, file_name="chat_history.txt")