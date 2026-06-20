import os
import shutil
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
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

# This reads your custom image from your folder
bg_base64 = get_base64_image("background.jpg")

glass_and_animation_css = f"""
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

/* Custom Matrix Pulse Animation for Uploading */
@keyframes matrixGlow {{
    0% {{ box-shadow: 0 0 10px rgba(59, 130, 246, 0.2); border-color: rgba(255, 255, 255, 0.15); }}
    50% {{ box-shadow: 0 0 30px rgba(59, 130, 246, 0.6); border-color: rgba(59, 130, 246, 0.6); }}
    100% {{ box-shadow: 0 0 10px rgba(59, 130, 246, 0.2); border-color: rgba(255, 255, 255, 0.15); }}
}}

.uploading-active {{
    animation: matrixGlow 2s infinite ease-in-out;
    background: rgba(17, 34, 64, 0.7) !important;
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
st.markdown(glass_and_animation_css, unsafe_allow_html=True)

# --- 2. SESSION INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# --- 3. TOP MAIN HEADER ---
st.markdown('''
<div style="text-align: center; margin-bottom: 40px; margin-top: 15px;">
    <h1 style="font-size: 3.2rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 40%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px;">
        ⚡ CoreIntellex Engine
    </h1>
    <p style="font-size: 1.15rem; color: rgba(255, 255, 255, 0.65) !important; font-weight: 300;">
        Deep Document Intelligence. Transform static PDFs into conversational vector fields right on your local architecture.
    </p>
</div>
''', unsafe_allow_html=True)

# --- 4. PERFECT TWO-BAR SPLIT COLUMN LAYOUT ---
left_bar, right_bar = st.columns([1, 1.4], gap="large")

# --- LEFT BAR: PERSISTENT UPLOADER CONTROL ---
with left_bar:
    # Applying regular panel styling or pulsing animation class based on active process state
    panel_class = "uploading-active" if st.session_state.is_processing else ""
    
    st.markdown(f'''
    <div class="glass-panel {panel_class}">
        <h3 style="margin-top:0; font-size:1.4rem; font-weight:700;">📥 Document Vault</h3>
        <p style="font-size:0.9rem; color:rgba(255,255,255,0.6)!important; margin-bottom:15px;">
            Drag and drop target records to dynamically compile structural knowledge graphs.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Control Target", type=["pdf"], label_visibility="collapsed")
    
    # Trigger processing only if file is fresh and different
    if uploaded_file and st.session_state.current_file != uploaded_file.name:
        st.session_state.is_processing = True
        
        with st.status("Initializing Vector Stream...", expanded=True) as status:
            time.sleep(0.5) # Fluid animation anchor
            
            st.write("📂 Stage 1: Parsing file contents into local directory...")
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write("✂️ Stage 2: Fragmenting structural text blocks...")
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            st.write("🧠 Stage 3: Embedding deep local neural vector arrays...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            db_directory = "chroma_db"
            if os.path.exists(db_directory):
                try: shutil.rmtree(db_directory)
                except Exception: pass
                
            vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_directory)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            
            st.write("🤖 Stage 4: Hot-linking context structures to local Llama 3.2...")
            llm = ChatOllama(model="llama3.2", temperature=0.2)
            
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
            
            st.session_state.current_file = uploaded_file.name
            st.session_state.chat_history = []  # Clear history for fresh document context
            st.session_state.is_processing = False
            status.update(label="✅ Data Matrix Indexed Successfully!", state="complete", expanded=False)
            st.rerun()

    # If document has been successfully indexed, present status badge below uploader element
    if st.session_state.current_file:
        st.markdown(f'''
        <div class="glass-panel" style="border-left: 4px solid #10b981!important; padding: 20px !important;">
            <p style="margin:0; font-size:0.8rem; text-transform:uppercase; color:#10b981!important; font-weight:700; letter-spacing:1px;">Active Memory Context</p>
            <h4 style="margin:5px 0 0 0; color:white; font-size:1.1rem; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">📄 {st.session_state.current_file}</h4>
        </div>
        ''', unsafe_allow_html=True)

# --- RIGHT BAR: DEDICATED AI SEARCH & CHAT INTERFACE ---
with right_bar:
    if st.session_state.rag_chain:
        st.markdown('''
        <div class="glass-panel" style="padding:15px 25px !important; margin-bottom:20px;">
            <h3 style="margin:0; font-size:1.3rem; font-weight:700; display:flex; align-items:center; gap:10px;">
                <span>🧠 Neural Search Interface</span>
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
        if user_query := st.chat_input("Query target document vector matrix..."):
            with chat_canvas:
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Decoding local node clusters..."):
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
    else:
        # Placeholder view block when database fields are empty
        st.markdown('''
        <div class="glass-panel" style="text-align:center; padding:100px 30px !important; opacity:0.85;">
            <div style="font-size:3.5rem; margin-bottom:15px;">⚡</div>
            <h4 style="color:rgba(255,255,255,0.7)!important; font-weight:500;">Awaiting Vector Initialization</h4>
            <p style="font-size:0.95rem; color:rgba(255,255,255,0.4)!important; max-width:400px; margin:5px auto 0 auto;">
                Please drag and drop a document into the left vault panel to open the encrypted context chat layers.
            </p>
        </div>
        ''', unsafe_allow_html=True)