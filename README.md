# 🍽️ RecipeGenie AI – Intelligent Recipe Generator & Document Q&A Assistant

> **A multi-agent AI cooking assistant powered by IBM watsonx.ai Granite Models, LangChain RAG, FAISS Vector DB, and Flask.**

---

## 🏗️ Architecture Overview

```
User (Browser)
    │
    ▼
Flask API (app.py)          ← Single-file application
    ├── Agent 1: Recipe Knowledge Agent
    │       ├── FAISS Vector Store (in-memory)
    │       ├── HuggingFace MiniLM-L6 Embeddings
    │       ├── LangChain RetrievalQA Chain
    │       └── IBM Granite 13B Chat v2  ──► IBM watsonx.ai Cloud
    │
    └── Agent 2: Personalized Recipe Generator
            └── IBM Granite 13B Instruct v2  ──► IBM watsonx.ai Cloud
```

---

## 🤖 AI Agents

### Agent 1 – Recipe Knowledge Agent (RAG)
Answers cooking and recipe questions by searching your uploaded documents using **Retrieval-Augmented Generation (RAG)**.

| Component | Technology |
|-----------|-----------|
| LLM | `ibm/granite-13b-chat-v2` via IBM watsonx.ai |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (in-memory) |
| Framework | LangChain `RetrievalQA` with custom prompt |
| Chunk size | 800 chars / 120 overlap |
| Top-K retrieval | 4 chunks per query |

**Example questions:**
- How do I make authentic vegetable biryani?
- What is the difference between baking soda and baking powder?
- What are common substitutes for eggs in baking?
- Which flour is best for gluten-free cakes?
- How do I make sugar-free chocolate cake?

### Agent 2 – Personalized Recipe Generator
Generates complete recipes tailored to user constraints using IBM Granite's instruction-following capability.

| Input Field | Options |
|-------------|---------|
| Ingredients | Free text (comma-separated) |
| Meal Type | Breakfast, Lunch, Dinner, Snack, Dessert, Appetiser |
| Cuisine | Any, Indian, Italian, Mexican, Chinese, Mediterranean, American, French, Japanese, Middle Eastern, Thai |
| Cooking Time | 15 / 30 / 45 / 60 / 90 / 120+ minutes |
| Difficulty | Easy / Medium / Hard |
| Servings | 1 / 2 / 4 / 6 / 8 / 10+ |
| Dietary | None / Vegetarian / Vegan / Gluten-Free / Keto / Paleo / Dairy-Free / Low-Carb / Low-Calorie / High-Protein |
| Allergies | Free text |
| Special Notes | Free text |

**Built-in presets:** High-Protein Breakfast, Vegan Pasta, Keto Dessert, Indian Dinner

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- IBM Cloud account with **watsonx.ai** service
- A watsonx.ai **Project ID**
- An IBM Cloud **API Key**

### 2. Clone / Download
```bash
# Place app.py, requirements.txt, and .env.example in the same folder
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
# Copy the example and fill in your values
copy .env.example .env      # Windows
# or
cp .env.example .env        # macOS/Linux
```

Edit `.env`:
```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 5. Run
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
.
├── app.py              ← Complete single-file application
├── requirements.txt    ← Python dependencies
├── .env.example        ← Credential template
├── .env                ← Your credentials (never commit this!)
└── uploads/            ← Auto-created, stores uploaded documents
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/api/config-status` | Check credentials & system status |
| `POST` | `/api/upload` | Upload a recipe document (PDF/TXT/MD) |
| `GET` | `/api/documents` | List all uploaded documents |
| `DELETE` | `/api/documents/<id>` | Remove a document from the knowledge base |
| `POST` | `/api/agent1/ask` | Agent 1: RAG Q&A query |
| `POST` | `/api/agent2/generate` | Agent 2: Recipe generation |

### POST `/api/agent1/ask`
```json
{ "question": "How do I make authentic vegetable biryani?" }
```
Response:
```json
{
  "answer": "...",
  "source": "IBM Granite + RAG (uploaded documents)",
  "chunks_used": 4
}
```

### POST `/api/agent2/generate`
```json
{
  "ingredients": "eggs, oats, banana",
  "meal_type": "Breakfast",
  "cuisine": "Any",
  "cooking_time": "15",
  "difficulty": "Easy",
  "servings": "2",
  "dietary": "High-Protein",
  "allergies": "none",
  "notes": "Quick morning meal"
}
```
Response:
```json
{
  "recipe": "🍽️ Protein-Packed Banana Oat Pancakes\n...",
  "model": "ibm/granite-13b-instruct-v2",
  "preferences_used": { ... }
}
```

---

## 🧠 How RAG Works in this App

1. **Upload** – User uploads a PDF or text recipe file
2. **Chunking** – Document split into 800-char chunks (120-char overlap) via `RecursiveCharacterTextSplitter`
3. **Embedding** – Each chunk is embedded using `all-MiniLM-L6-v2` (512-dim vectors)
4. **Indexing** – Vectors stored in a FAISS in-memory index
5. **Retrieval** – At query time, top-4 most similar chunks are retrieved
6. **Augmentation** – Retrieved chunks injected into IBM Granite's context window
7. **Generation** – Granite generates an answer grounded in your actual documents

---

## 🏷️ IBM Granite Models

| Model | Use Case | Temperature |
|-------|----------|-------------|
| `ibm/granite-13b-chat-v2` | RAG-based Q&A (Agent 1) | 0.5 |
| `ibm/granite-13b-instruct-v2` | Recipe generation (Agent 2) | 0.75 |

Max new tokens: 900 (Agent 1) / 1400 (Agent 2)

---

## 🌐 Supported Document Formats

| Format | Support |
|--------|---------|
| `.pdf` | ✅ via PyPDF2 |
| `.txt` | ✅ plain text |
| `.md` | ✅ Markdown |
| Max size | 16 MB |

---

## 🔒 Security Notes

- Never commit `.env` to version control — add it to `.gitignore`
- API keys shown in the Config panel are only displayed client-side; they are **not sent back to the server**
- Uploaded files are stored locally in `uploads/` and never sent to third parties beyond watsonx.ai inference

---

## 📦 Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, Flask 3 |
| AI / LLM | IBM watsonx.ai, IBM Granite 13B |
| RAG Framework | LangChain, LangChain-IBM |
| Vector Database | FAISS |
| Embeddings | HuggingFace sentence-transformers |
| PDF Parsing | PyPDF2 |
| Frontend | HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons, Vanilla JS |
| Config | python-dotenv |

---

*RecipeGenie AI – IBM watsonx.ai Granite Demo*
