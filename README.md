# 🤖 SmartDocumentBot – Your AI-Powered Document Assistant

SmartDocuBot is an interactive chatbot that lets users upload documents (PDF, DOCX, TXT), processes them with LLMs, and allows natural language queries to extract relevant answers. Built using **LangChain**, **FAISS**, **SentenceTransformers**, and **Groq’s LLMs**, all inside a stylish **Streamlit UI**.

---

## 🚀 Features

- 📁 Upload PDFs, DOCX, or TXT files
- 🧠 Preprocess & embed content using `RecursiveCharacterTextSplitter` and `SentenceTransformer`
- 🔍 Create FAISS vector store for efficient semantic search
- 🤖 Ask questions and get accurate responses from **Groq’s LLM** (Gemma replacement model)
- 💬 Sleek chatbot UI with:
  - Sidebar upload & preprocess
  - Query input with "Generate" button
  - Auto-cleared input field after every query
  - Chat history download option
- 🎨 Multi-theme and creative CSS for a clean UX

---

## 🛠️ Tech Stack

| Layer            | Tech Used                              |
|------------------|----------------------------------------|
| 💬 LLM           | Groq (`llama3-8b-8192`)                 |
| 📚 LangChain     | PromptTemplate, RetrievalQA, Chains     |
| 🔎 Embedding     | `sentence-transformers/all-MiniLM-L6-v2`|
| 📊 Vector DB     | FAISS                                   |
| 🖥️ Frontend      | Streamlit + Custom CSS                  |
| 📄 File Handling | PyPDFLoader, python-docx, plain text        |

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/SmartDocuBot.git
cd SmartDocuBot
