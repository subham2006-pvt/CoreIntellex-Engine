# ⚡ CoreIntellex Engine

A premium, production-grade local Retrieval-Augmented Generation (RAG) pipeline featuring a breathtaking **Liquid Glassmorphism (Glass-UI)** design. Built entirely on local architecture, it converts static PDF records into searchable multi-turn vector spaces—fully private, secure, and offline.

---

## ✨ Features
* **🔮 High-End Liquid Glassmorphism UI:** Built on Streamlit, leveraging custom injected CSS keyframe animations, blurred alpha backdrops, and custom visual layers.
* **📥 Active Matrix File Processing:** Multi-stage vector initialization handles dynamic file parsing, semantic recursive chunking, and memory synchronization.
* **🧠 History-Aware Conversational Threading:** Remembers prior contextual blocks, seamlessly resolving ambiguous pronouns (e.g., *"What is its largest city?"* following an initial query about Odisha).
* **🔒 100% Privacy-Preserved:** Powered completely by your local machine using local embedding matrices (`all-MiniLM-L6-v2`) and local LLMs (`Llama 3.2`).

---

## 🛠️ Architecture Stack
* **Frontend Interface:** Streamlit (Custom Glass Injection Layouts)
* **Orchestration Framework:** LangChain / LangChain-Community
* **Vector Engine Database:** Chroma DB (Locally Parsed Indices)
* **Local Inference Provider:** Ollama (`llama3.2`)
* **Embedding Pipeline:** HuggingFace `all-MiniLM-L6-v2`

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have the [Ollama App](https://ollama.com) installed and the model downloaded locally:
```bash
ollama pull llama3.2