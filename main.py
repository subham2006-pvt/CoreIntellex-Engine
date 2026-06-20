import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def main():
    # --- STEP 4: Load PDF Document ---
    print("\n--- Step 4: Loading PDF Document ---")
    pdf_path = os.path.join("data", "Odisha.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find 'Odisha.pdf' inside the 'data/' folder.")
        print("Please verify that your PDF is sitting inside your 'data' folder and named exactly 'Odisha.pdf'")
        return
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Success: Loaded {len(docs)} pages from the PDF document.")

    # --- STEP 5: Split Text into Chunks ---
    print("\n--- Step 5: Splitting Text into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Success: Divided document into {len(chunks)} total text chunks.")

    # --- STEP 6: Generate Embeddings ---
    print("\n--- Step 6: Generating Embeddings ---")
    # This model runs 100% locally on your Mac CPU/GPU
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"✅ Success: Local embedding model loaded successfully.")

    # --- STEP 7: Create Vector Database ---
    print("\n--- Step 7: Creating Vector Database ---")
    db_directory = "chroma_db"
    
    # Optional cleanup: Clear old database folders if they exist to prevent duplication issues
    if os.path.exists(db_directory):
        try:
            shutil.rmtree(db_directory)
        except Exception:
            pass

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=db_directory
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Success: Vector database created and indexed locally at './{db_directory}'.")

    # --- STEP 8-12: Answer Generation Setup ---
    print("\n🤖 Connecting to local Llama 3.2...")
    try:
        # Corrected model tag string to match your exact local Ollama image name
        llm = ChatOllama(model="llama3.2", temperature=0.2)
    except Exception as e:
        print("❌ Error: Could not connect to Ollama. Ensure the Ollama app is open on your Mac!")
        return

    system_prompt = (
        "You are an expert assistant specialized in answering questions using provided documents.\n"
        "Answer the user's question using ONLY the retrieved context blocks below. "
        "If the answer cannot be found in the context, politely say: 'I cannot find that information in the provided document.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n🚀 Local AnswerBot RAG Pipeline is completely live and ready!")
    print("Type 'exit' to close.")
    
    while True:
        user_question = input("\n👉 Enter your question about Odisha: ")
        if user_question.strip().lower() == 'exit':
            print("Goodbye!")
            break
            
        if not user_question.strip():
            continue

        print("🔍 Searching local context and generating answer...")
        try:
            response = rag_chain.invoke({"input": user_question})
            print("\n🤖 Answer:")
            print("=" * 60)
            print(response["answer"])
            print("=" * 60)
        except Exception as e:
            print(f"❌ Generation error: {e}")

if __name__ == "__main__":
    main()