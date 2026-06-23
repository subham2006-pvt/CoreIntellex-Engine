import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. VIEWPORT LAYOUT & STYLING INJECTION ---
st.set_page_config(page_title="CoreIntellex Engine", layout="wide", page_icon="⚡")

import base64

def get_base64_image(file_path):
    """Converts local image to base64 for smooth background injection."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

# Reads custom background image asset
bg_base64 = get_base64_image("background.jpg")

glass_css = f"""
<style>
/* Background Image Configuration */
.stApp {{
    background: url("data:image/jpg;base64,{bg_base64}") no-repeat center center fixed !important;
    background-size: cover !important;
}}

/* Liquid Glass Panels Styling */
.glass-panel {{
    background: rgba(13, 20, 38, 0.55) !important;
    backdrop-filter: blur(25px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(25px) saturate(140%) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6) !important;
    padding: 30px !important;
    margin-bottom: 20px;
}}

h1, h2, h3, h4, p, span, label, .stMarkdown {{
    color: #ffffff !important;
    font-family: 'Inter', -apple-system, sans-serif;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
}}

/* Clean Custom Input Bar styles */
.stChatInputContainer {{
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
}}
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# --- 2. SESSION INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- 3. TOP MAIN HEADER ---
st.markdown('''
<div style="text-align: center; margin-bottom: 40px; margin-top: 15px;">
    <h1 style="font-size: 3.2rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 40%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px;">
        ⚡ CoreIntellex Engine
    </h1>
    <p style="font-size: 1.15rem; color: rgba(255, 255, 255, 0.65) !important; font-weight: 300;">
        Deep Document Intelligence. Dynamic vector space powered fully via pre-compiled matrices.
    </p>
</div>
''', unsafe_allow_html=True)

# --- 4. BACKEND COUPLING (AUTO-LOAD VECTOR DATABASE) ---
DB_DIR = "chroma_db"

if st.session_state.rag_chain is None:
    # Safety Check: Verify database exists
    if not os.path.exists(DB_DIR):
        st.error(f"❌ Error: Database directory `{DB_DIR}` not found! Run your data processing script locally first to generate vectors.")
        st.stop()
        
    # Get Groq Key from Streamlit Secrets or Environment Variable
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY configuration missing! Please configure it in your Streamlit Advanced Settings secrets container.")
        st.stop()
        
    # Wire directly into the static local vector database
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Using cloud LLM infrastructure for safe production deployments
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=groq_api_key)
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    system_prompt = (
        "You are an expert assistant specialized in answering questions using provided documents.\n"
        "Answer the user's question using ONLY the retrieved context blocks below.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 5. CLEAN CENTRALIZED SEARCH INTERFACE ---
left_spacer, center_workspace, right_spacer = st.columns([0.2, 2, 0.2])

with center_workspace:
    st.markdown('''
    <div class="glass-panel" style="padding:15px 25px !important; margin-bottom:20px;">
        <h3 style="margin:0; font-size:1.3rem; font-weight:700; display:flex; align-items:center; gap:10px;">
            <span>🧠 Active Knowledge Stream Interface</span>
        </h3>
    </div>
    ''', unsafe_allow_html=True)
    
    # Continuous canvas container for dialogue lines
    chat_canvas = st.container()
    with chat_canvas:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)

    # Bottom anchored text-bar parsing user prompts
    if user_query := st.chat_input("Ask anything contained within the core document matrix..."):
        with chat_canvas:
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Decoding vector node records..."):
                    response = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    st.write(answer)
                    
        # Save query loop directly inside persistent history states
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=answer)
        ])