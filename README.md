# ⚖️ LexiQuery — Intelligent Legal Contract Analyzer

> AI-powered legal contract analysis system built with RAG architecture, ChromaDB, Groq (Llama 3.3 70B), and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-green)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What is LexiQuery?

LexiQuery is a **full stack Generative AI application** that uses 
**Retrieval Augmented Generation (RAG)** to help non-lawyers 
understand legal contracts instantly.

Upload any legal contract → Ask questions in plain English → 
Get AI-powered answers with clause citations and risk scores.

---

## ✨ Features

- 📄 **Smart PDF Parsing** — Extracts and chunks contracts by legal clauses
- 💬 **Natural Language Q&A** — Ask any question about your contract
- 🎯 **Risk Scoring** — Automatic 1-10 risk score with red flags
- 🔍 **Multi-Contract Comparison** — Compare contracts side by side
- 📎 **Source Citations** — Every answer cites the exact clause
- 🚀 **Fast** — Powered by Groq's ultra-fast Llama 3.3 70B

---

## 🏗️ Architecture
```
User uploads PDF
      ↓
PDF Parser (pdfplumber)
      ↓
Clause Chunker (regex-based legal structure detection)
      ↓
Embeddings (sentence-transformers: all-MiniLM-L6-v2)
      ↓
Vector Store (ChromaDB — persistent local storage)
      ↓
User asks question
      ↓
Semantic Retrieval (cosine similarity search)
      ↓
Context + Question → Groq LLM (Llama 3.3 70B)
      ↓
Plain English Answer with clause citations
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| PDF Parsing | pdfplumber |
| Frontend | Streamlit |
| Backend | FastAPI |
| Containerization | Docker |
| Version Control | Git + GitHub |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Groq API key (free at console.groq.com)

### Installation
```bash
# Clone the repository
git clone https://github.com/Ronilmuchandi/lexiquery.git
cd lexiquery

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env file
```

### Run the App
```bash
PYTHONPATH=/path/to/lexiquery streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### Docker
```bash
# Build
docker build -t lexiquery .

# Run
docker run -p 8501:8501 --env-file .env lexiquery
```

---

## 📁 Project Structure
```
lexiquery/
├── app/
│   ├── main.py              # FastAPI backend
│   └── streamlit_app.py     # Streamlit frontend
├── src/
│   ├── llm/
│   │   └── groq_client.py   # Groq LLM connection
│   ├── rag/
│   │   ├── parser.py        # PDF text extraction
│   │   ├── chunker.py       # Clause-level chunking
│   │   ├── embedder.py      # Sentence embeddings
│   │   └── retriever.py     # ChromaDB operations
│   └── pipeline/
│       ├── indexer.py       # Contract indexing pipeline
│       ├── analyzer.py      # RAG Q&A pipeline
│       └── comparator.py    # Risk scoring + comparison
├── data/
│   └── contracts/           # Sample contracts
├── config/                  # Configuration files
├── Dockerfile               # Container setup
├── requirements.txt         # Dependencies
└── README.md                # You are here!
```

---

## 💡 Example Queries

- *"What are my obligations if I want to terminate this agreement?"*
- *"Are there any non-compete clauses? What do they cover?"*
- *"Flag any clauses that are unusual or particularly risky"*
- *"What information is considered confidential?"*

---

## 📊 Sample Output
```
Question: What are my obligations if I terminate this agreement?

Answer: If you want to terminate this agreement, you must:
1. Continue keeping all Confidential Information secret (Clause 4)
2. Return any records or materials to the Disclosing Party (Clause 3)
3. Note that confidentiality obligations survive termination (Clause 4)

Risk Score: 2/10 — Very Low Risk ✅
Sources: Clause 3, Clause 4 from NDA_Basic
```

---

## 🔮 Future Improvements

- [ ] OCR support for scanned PDFs
- [ ] Multi-language contract support  
- [ ] Export analysis as PDF report
- [ ] Clause-by-clause walkthrough mode
- [ ] Integration with cloud storage (AWS S3)

---

## 👨‍💻 Author

**Ronil Muchandi**  
MS Data Science & Analytics — University of Missouri  
📧 ronilmizzou@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/ronil-muchandi-892602187)  
🐙 [GitHub](https://github.com/Ronilmuchandi)

---

## ⚠️ Disclaimer

LexiQuery is for informational purposes only and does not 
constitute legal advice. Always consult a qualified lawyer 
for important legal decisions.