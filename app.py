"""
RecipeGenie AI – Intelligent Recipe Generator & Document Q&A Assistant
======================================================================
Single-file Flask application powered by IBM watsonx.ai Granite Models,
LangChain RAG pipeline, and a FAISS vector database.

Agents
------
  Agent 1 – Recipe Knowledge Agent   : RAG-based Q&A over uploaded recipe PDFs / text files
  Agent 2 – Personalized Recipe Gen  : Constraint-driven recipe generation

Dependencies (install before running):
    pip install flask python-dotenv ibm-watsonx-ai langchain langchain-community
                langchain-ibm faiss-cpu pypdf2 tiktoken sentence-transformers

Environment variables (.env):
    WATSONX_API_KEY=<your IBM Cloud API key>
    WATSONX_PROJECT_ID=<your watsonx.ai project id>
    WATSONX_URL=https://us-south.ml.cloud.ibm.com
"""

import os
import io
import json
import uuid
import textwrap
import traceback
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify, session,
    render_template_string, send_from_directory
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ── LangChain / IBM watsonx imports ──────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", uuid.uuid4().hex)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = Path("uploads")
VECTOR_FOLDER   = Path("vector_store")
ALLOWED_EXT     = {"pdf", "txt", "md"}
MAX_CONTENT_MB  = 16

UPLOAD_FOLDER.mkdir(exist_ok=True)
VECTOR_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"]    = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

# IBM Granite model IDs
GRANITE_CHAT_MODEL  = "ibm/granite-13b-chat-v2"
GRANITE_INST_MODEL  = "ibm/granite-13b-instruct-v2"

# In-memory vector store cache
_vector_store: FAISS | None = None
_uploaded_docs: list[dict]  = []

# ── Watsonx LLM factory ───────────────────────────────────────────────────────
def _make_llm(model_id: str, max_tokens: int = 1024, temperature: float = 0.7) -> WatsonxLLM:
    """Return a LangChain-compatible WatsonxLLM instance."""
    params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.TEMPERATURE:    temperature,
        GenParams.TOP_P:          0.9,
        GenParams.TOP_K:          50,
        GenParams.REPETITION_PENALTY: 1.1,
    }
    return WatsonxLLM(
        model_id=model_id,
        url=WATSONX_URL,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        params=params,
    )


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Document helpers ──────────────────────────────────────────────────────────
def _extract_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        if not PDF_SUPPORT:
            return "[PDF support unavailable – install PyPDF2]"
        text = []
        with open(filepath, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _rebuild_vector_store() -> None:
    global _vector_store
    if not _uploaded_docs:
        _vector_store = None
        return
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "]
    )
    all_chunks = []
    for doc in _uploaded_docs:
        chunks = splitter.split_text(doc["text"])
        all_chunks.extend(chunks)
    if not all_chunks:
        _vector_store = None
        return
    emb = _embeddings()
    _vector_store = FAISS.from_texts(all_chunks, emb)


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Agent 1 – Recipe Knowledge Agent (RAG) ───────────────────────────────────
_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""
        You are RecipeGenie's expert culinary knowledge assistant powered by IBM Granite.
        Use ONLY the information from the provided recipe documents to answer the question.
        If the documents do not contain enough information, say so clearly, then offer
        brief general culinary guidance.

        --- RECIPE DOCUMENT CONTEXT ---
        {context}
        --- END CONTEXT ---

        Question: {question}

        Provide a detailed, helpful, and accurate answer. Format your response with clear
        sections (e.g., Ingredients, Steps, Tips) when applicable.
        Answer:
    """).strip(),
)


def agent1_rag_answer(question: str) -> dict:
    """Agent 1: RAG-powered recipe Q&A."""
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return {"error": "IBM watsonx.ai credentials are not configured."}

    if _vector_store is None:
        # No documents – use pure LLM
        llm = _make_llm(GRANITE_CHAT_MODEL, max_tokens=900, temperature=0.6)
        fallback_prompt = (
            "You are RecipeGenie, an expert culinary AI assistant powered by IBM Granite.\n"
            f"Answer the following cooking question thoroughly:\n\n{question}\n\nAnswer:"
        )
        answer = llm.invoke(fallback_prompt)
        return {
            "answer": answer,
            "source": "IBM Granite (no documents uploaded – general knowledge)",
            "chunks_used": 0,
        }

    llm = _make_llm(GRANITE_CHAT_MODEL, max_tokens=900, temperature=0.5)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": _RAG_PROMPT},
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "source": "IBM Granite + RAG (uploaded documents)",
        "chunks_used": len(result.get("source_documents", [])),
    }


# ── Agent 2 – Personalized Recipe Generator ──────────────────────────────────
_RECIPE_GEN_TEMPLATE = textwrap.dedent("""
    You are RecipeGenie's personalized recipe creation agent powered by IBM Granite.
    Generate a complete, detailed, and delicious recipe based on the user's preferences below.

    USER PREFERENCES:
    - Ingredients available : {ingredients}
    - Dietary restrictions  : {dietary}
    - Cuisine preference    : {cuisine}
    - Meal type             : {meal_type}
    - Cooking time          : {cooking_time} minutes
    - Difficulty level      : {difficulty}
    - Number of servings    : {servings}
    - Allergies             : {allergies}
    - Special notes         : {notes}

    Generate a complete recipe with:
    1. 🍽️ Recipe Name & Brief Description
    2. ⏱️ Prep Time | Cook Time | Total Time | Servings
    3. 🥗 Nutrition Highlights (calories estimate, protein, key nutrients)
    4. 📋 Ingredients (with exact quantities)
    5. 👨‍🍳 Step-by-Step Instructions (numbered, clear)
    6. 💡 Chef's Tips & Tricks
    7. 🔄 Ingredient Substitutions (for common allergens or dietary needs)
    8. 🍱 Storage & Meal Prep Advice

    Make the recipe practical, delicious, and perfectly tailored to the preferences above.
    Recipe:
""").strip()


def agent2_generate_recipe(prefs: dict) -> dict:
    """Agent 2: Personalized recipe generation."""
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return {"error": "IBM watsonx.ai credentials are not configured."}

    prompt = _RECIPE_GEN_TEMPLATE.format(
        ingredients  = prefs.get("ingredients", "pantry staples"),
        dietary      = prefs.get("dietary", "none"),
        cuisine      = prefs.get("cuisine", "any"),
        meal_type    = prefs.get("meal_type", "any"),
        cooking_time = prefs.get("cooking_time", "30"),
        difficulty   = prefs.get("difficulty", "medium"),
        servings     = prefs.get("servings", "2"),
        allergies    = prefs.get("allergies", "none"),
        notes        = prefs.get("notes", "none"),
    )

    llm = _make_llm(GRANITE_INST_MODEL, max_tokens=1400, temperature=0.75)
    recipe_text = llm.invoke(prompt)
    return {
        "recipe": recipe_text,
        "model": GRANITE_INST_MODEL,
        "preferences_used": prefs,
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/config-status")
def config_status():
    configured = bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)
    return jsonify({
        "configured": configured,
        "model_chat": GRANITE_CHAT_MODEL,
        "model_inst": GRANITE_INST_MODEL,
        "docs_loaded": len(_uploaded_docs),
        "vector_store_ready": _vector_store is not None,
    })


@app.route("/api/upload", methods=["POST"])
def upload_document():
    global _uploaded_docs
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": f"File type not allowed. Supported: {ALLOWED_EXT}"}), 400

    filename  = secure_filename(file.filename)
    unique    = f"{uuid.uuid4().hex}_{filename}"
    save_path = UPLOAD_FOLDER / unique
    file.save(str(save_path))

    text = _extract_text(str(save_path))
    if not text.strip():
        return jsonify({"error": "Could not extract text from file"}), 400

    doc_entry = {
        "id":        unique,
        "name":      filename,
        "path":      str(save_path),
        "text":      text,
        "size":      save_path.stat().st_size,
        "uploaded":  datetime.now().isoformat(),
        "chars":     len(text),
    }
    _uploaded_docs.append(doc_entry)
    _rebuild_vector_store()

    return jsonify({
        "success":   True,
        "doc_id":    unique,
        "name":      filename,
        "chars":     len(text),
        "total_docs": len(_uploaded_docs),
        "vector_ready": _vector_store is not None,
    })


@app.route("/api/documents")
def list_documents():
    docs = [
        {"id": d["id"], "name": d["name"], "chars": d["chars"], "uploaded": d["uploaded"]}
        for d in _uploaded_docs
    ]
    return jsonify({"documents": docs, "count": len(docs)})


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    global _uploaded_docs
    before = len(_uploaded_docs)
    _uploaded_docs = [d for d in _uploaded_docs if d["id"] != doc_id]
    if len(_uploaded_docs) == before:
        return jsonify({"error": "Document not found"}), 404
    _rebuild_vector_store()
    return jsonify({"success": True, "remaining": len(_uploaded_docs)})


@app.route("/api/agent1/ask", methods=["POST"])
def agent1_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        result = agent1_rag_answer(question)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/agent2/generate", methods=["POST"])
def agent2_generate():
    data = request.get_json(silent=True) or {}
    try:
        result = agent2_generate_recipe(data)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ── HTML Template (single-file embedded frontend) ────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RecipeGenie AI – IBM watsonx.ai</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"/>
<style>
  :root {
    --ibm-blue:#0f62fe; --ibm-dark:#001141; --ibm-light:#edf5ff;
    --granite:#6929c4;  --accent:#ff832b;   --success:#198038;
    --card-radius:12px; --shadow:0 2px 12px rgba(0,0,0,.08);
  }
  *{box-sizing:border-box;}
  body{background:#f4f6fb;font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
       color:#1c1c1e;min-height:100vh;}

  /* ── Navbar ── */
  .navbar{background:linear-gradient(135deg,var(--ibm-dark) 0%,#0f3460 100%);
          box-shadow:0 2px 16px rgba(0,0,0,.3);}
  .navbar-brand{font-size:1.4rem;font-weight:700;color:#fff!important;letter-spacing:-.3px;}
  .navbar-brand span{color:var(--accent);}
  .badge-ibm{background:var(--ibm-blue);font-size:.65rem;vertical-align:middle;
             border-radius:4px;padding:2px 6px;}
  .badge-granite{background:var(--granite);font-size:.65rem;vertical-align:middle;
                 border-radius:4px;padding:2px 6px;}
  .status-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;}
  .dot-ok{background:#42be65;} .dot-warn{background:#f1c21b;} .dot-err{background:#fa4d56;}

  /* ── Sidebar ── */
  .sidebar{width:240px;min-height:calc(100vh - 60px);background:#fff;
           border-right:1px solid #e0e0e0;padding:1.2rem .8rem;position:sticky;top:60px;}
  .nav-section{font-size:.68rem;font-weight:700;color:#6f6f6f;text-transform:uppercase;
               letter-spacing:.08em;padding:.6rem .5rem .2rem;}
  .sidebar .nav-link{border-radius:8px;color:#3d3d3d;font-size:.9rem;padding:.45rem .8rem;
                     display:flex;align-items:center;gap:.5rem;transition:all .15s;}
  .sidebar .nav-link:hover,.sidebar .nav-link.active{background:var(--ibm-light);
    color:var(--ibm-blue);font-weight:600;}
  .sidebar .nav-link i{width:18px;text-align:center;}

  /* ── Cards & Panels ── */
  .panel{display:none;animation:fadeIn .25s ease;}
  .panel.active{display:block;}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

  .card{border:none;border-radius:var(--card-radius);box-shadow:var(--shadow);}
  .card-header-ibm{background:linear-gradient(135deg,var(--ibm-blue),#0043ce);
                   color:#fff;border-radius:var(--card-radius) var(--card-radius) 0 0!important;
                   padding:.9rem 1.2rem;font-weight:600;}
  .card-header-granite{background:linear-gradient(135deg,var(--granite),#491d8b);
                       color:#fff;border-radius:var(--card-radius) var(--card-radius) 0 0!important;
                       padding:.9rem 1.2rem;font-weight:600;}

  /* ── Hero ── */
  .hero{background:linear-gradient(135deg,var(--ibm-dark) 0%,#0f3460 60%,var(--granite) 100%);
        color:#fff;border-radius:var(--card-radius);padding:2.5rem 2rem;margin-bottom:1.5rem;
        position:relative;overflow:hidden;}
  .hero::after{content:"";position:absolute;right:-60px;top:-60px;width:280px;height:280px;
               border-radius:50%;background:rgba(255,255,255,.04);}
  .hero h1{font-size:2rem;font-weight:800;margin-bottom:.4rem;}
  .hero p{opacity:.85;max-width:640px;font-size:.97rem;}
  .hero-badges{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem;}
  .hero-badge{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
              border-radius:20px;padding:.25rem .85rem;font-size:.78rem;}

  /* ── Agent cards (overview) ── */
  .agent-card{border-radius:var(--card-radius);padding:1.4rem;cursor:pointer;
              transition:transform .18s,box-shadow .18s;border:2px solid transparent;}
  .agent-card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(0,0,0,.12);}
  .agent-card.agent-1{background:linear-gradient(135deg,#edf5ff,#d0e2ff);border-color:#a6c8ff;}
  .agent-card.agent-2{background:linear-gradient(135deg,#f6f2ff,#e8daff);border-color:#d4bbff;}
  .agent-icon{font-size:2.2rem;margin-bottom:.6rem;}
  .agent-title{font-weight:700;font-size:1.05rem;margin-bottom:.3rem;}
  .agent-desc{font-size:.85rem;color:#555;line-height:1.5;}

  /* ── Upload zone ── */
  .upload-zone{border:2px dashed #a6c8ff;border-radius:10px;padding:2rem;
               text-align:center;background:#f0f6ff;cursor:pointer;transition:all .2s;}
  .upload-zone:hover,.upload-zone.drag{border-color:var(--ibm-blue);background:#e5f0ff;}
  .upload-zone i{font-size:2.5rem;color:#4589ff;margin-bottom:.5rem;}

  /* ── Chat bubble ── */
  .chat-wrap{max-height:520px;overflow-y:auto;padding:.5rem 0;display:flex;
             flex-direction:column;gap:.8rem;}
  .bubble{max-width:82%;border-radius:14px;padding:.75rem 1rem;line-height:1.55;
          font-size:.93rem;white-space:pre-wrap;word-break:break-word;}
  .bubble-user{background:var(--ibm-blue);color:#fff;align-self:flex-end;
               border-bottom-right-radius:4px;}
  .bubble-bot{background:#fff;border:1px solid #e0e0e0;align-self:flex-start;
              border-bottom-left-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.06);}
  .bubble-meta{font-size:.7rem;opacity:.65;margin-top:.3rem;}
  .bubble-error{background:#fff0f0;border:1px solid #ffb3b3;color:#a2191f;}

  /* ── Recipe output ── */
  .recipe-output{background:#fff;border-radius:10px;padding:1.5rem;
                 border:1px solid #e0e0e0;white-space:pre-wrap;line-height:1.65;
                 font-size:.93rem;max-height:680px;overflow-y:auto;}
  .recipe-output h3{color:var(--ibm-dark);font-size:1.1rem;margin-top:1rem;}

  /* ── Tags / pills ── */
  .tag{display:inline-block;background:#e8daff;color:var(--granite);
       border-radius:20px;padding:.18rem .65rem;font-size:.75rem;margin:.15rem;}

  /* ── Spinner ── */
  .thinking{display:flex;align-items:center;gap:.5rem;color:#6f6f6f;font-size:.85rem;
            padding:.5rem 1rem;}
  .dots span{display:inline-block;width:7px;height:7px;border-radius:50%;
             background:#4589ff;animation:bounce 1s infinite;}
  .dots span:nth-child(2){animation-delay:.15s;}
  .dots span:nth-child(3){animation-delay:.3s;}
  @keyframes bounce{0%,80%,100%{transform:scale(.8)}40%{transform:scale(1.2)}}

  /* ── Doc list ── */
  .doc-item{display:flex;justify-content:space-between;align-items:center;
            padding:.6rem .9rem;background:#f4f4f4;border-radius:8px;margin-bottom:.4rem;}
  .doc-item .doc-name{font-size:.88rem;font-weight:500;color:var(--ibm-dark);}
  .doc-item .doc-meta{font-size:.75rem;color:#6f6f6f;}

  /* ── Misc ── */
  .form-label{font-size:.85rem;font-weight:600;color:#393939;}
  .btn-ibm{background:var(--ibm-blue);border-color:var(--ibm-blue);color:#fff;
           border-radius:8px;font-weight:600;}
  .btn-ibm:hover{background:#0043ce;border-color:#0043ce;color:#fff;}
  .btn-granite{background:var(--granite);border-color:var(--granite);color:#fff;
               border-radius:8px;font-weight:600;}
  .btn-granite:hover{background:#491d8b;border-color:#491d8b;color:#fff;}
  .section-title{font-size:1.1rem;font-weight:700;color:var(--ibm-dark);
                 border-left:4px solid var(--ibm-blue);padding-left:.7rem;margin-bottom:1rem;}
  textarea.form-control{font-size:.9rem;}
  .rag-badge{background:#defbe6;color:var(--success);border-radius:6px;
             padding:.2rem .6rem;font-size:.72rem;font-weight:600;}
  .footer-bar{text-align:center;padding:1.2rem;color:#8d8d8d;font-size:.78rem;
              border-top:1px solid #e0e0e0;margin-top:2rem;}
</style>
</head>
<body>

<!-- ══ NAVBAR ══════════════════════════════════════════════════════════════ -->
<nav class="navbar navbar-expand-lg sticky-top">
  <div class="container-fluid px-3">
    <a class="navbar-brand" href="#">
      <i class="bi bi-stars me-1"></i>RecipeGenie <span>AI</span>
      <span class="badge-ibm ms-2">IBM watsonx.ai</span>
      <span class="badge-granite ms-1">Granite</span>
    </a>
    <div class="d-flex align-items-center gap-3 ms-auto">
      <div id="statusIndicator" class="d-flex align-items-center text-white-50 small">
        <span class="status-dot dot-warn" id="statusDot"></span>
        <span id="statusText">Checking...</span>
      </div>
      <span class="text-white-50 small d-none d-md-block">
        <i class="bi bi-cpu me-1"></i>Multi-Agent RAG System
      </span>
    </div>
  </div>
</nav>

<!-- ══ BODY LAYOUT ══════════════════════════════════════════════════════════ -->
<div class="d-flex" style="min-height:calc(100vh - 60px)">

  <!-- ── Sidebar ── -->
  <div class="sidebar d-none d-lg-block">
    <div class="nav-section">Navigation</div>
    <nav class="nav flex-column gap-1">
      <a class="nav-link active" onclick="showPanel('overview')" href="#">
        <i class="bi bi-grid-1x2"></i>Overview
      </a>
      <a class="nav-link" onclick="showPanel('agent1')" href="#">
        <i class="bi bi-search-heart"></i>Knowledge Agent
      </a>
      <a class="nav-link" onclick="showPanel('agent2')" href="#">
        <i class="bi bi-magic"></i>Recipe Generator
      </a>
    </nav>
    <div class="nav-section mt-3">Documents</div>
    <nav class="nav flex-column gap-1">
      <a class="nav-link" onclick="showPanel('docs')" href="#">
        <i class="bi bi-file-earmark-text"></i>Manage Docs
      </a>
    </nav>
    <div class="nav-section mt-3">System</div>
    <nav class="nav flex-column gap-1">
      <a class="nav-link" onclick="showPanel('config')" href="#">
        <i class="bi bi-gear"></i>Configuration
      </a>
    </nav>
    <div class="mt-4 p-2 rounded" style="background:#f4f4f4;font-size:.75rem;color:#555">
      <div class="fw-bold mb-1"><i class="bi bi-info-circle me-1"></i>RAG Pipeline</div>
      <div>Embeddings: <span class="fw-semibold">MiniLM-L6</span></div>
      <div>Vector DB: <span class="fw-semibold">FAISS</span></div>
      <div>LLM: <span class="fw-semibold">IBM Granite</span></div>
    </div>
  </div>

  <!-- ── Main content ── -->
  <div class="flex-grow-1 p-3 p-lg-4" style="max-width:1100px">

    <!-- Mobile nav pills -->
    <div class="d-lg-none mb-3 overflow-auto">
      <div class="d-flex gap-2 flex-nowrap">
        <button class="btn btn-sm btn-ibm" onclick="showPanel('overview')">
          <i class="bi bi-grid-1x2"></i> Overview</button>
        <button class="btn btn-sm btn-ibm" onclick="showPanel('agent1')">
          <i class="bi bi-search-heart"></i> Knowledge</button>
        <button class="btn btn-sm btn-granite" onclick="showPanel('agent2')">
          <i class="bi bi-magic"></i> Generator</button>
        <button class="btn btn-sm btn-outline-secondary" onclick="showPanel('docs')">
          <i class="bi bi-file-earmark-text"></i> Docs</button>
        <button class="btn btn-sm btn-outline-secondary" onclick="showPanel('config')">
          <i class="bi bi-gear"></i> Config</button>
      </div>
    </div>

    <!-- ════════ PANEL: OVERVIEW ════════ -->
    <div id="panel-overview" class="panel active">
      <div class="hero">
        <h1><i class="bi bi-stars me-2"></i>RecipeGenie AI</h1>
        <p>An intelligent multi-agent recipe assistant powered by <strong>IBM watsonx.ai Granite Models</strong>.
           Upload recipe documents, ask cooking questions using RAG, and generate fully personalised recipes.</p>
        <div class="hero-badges">
          <span class="hero-badge"><i class="bi bi-cpu me-1"></i>IBM Granite 13B</span>
          <span class="hero-badge"><i class="bi bi-search me-1"></i>RAG Pipeline</span>
          <span class="hero-badge"><i class="bi bi-database me-1"></i>FAISS Vector DB</span>
          <span class="hero-badge"><i class="bi bi-diagram-3 me-1"></i>Multi-Agent System</span>
          <span class="hero-badge"><i class="bi bi-cloud me-1"></i>IBM watsonx.ai</span>
        </div>
      </div>

      <div class="row g-3 mb-4">
        <div class="col-md-6">
          <div class="agent-card agent-1 h-100" onclick="showPanel('agent1')">
            <div class="agent-icon">🔍</div>
            <div class="agent-title">Agent 1 – Recipe Knowledge Agent</div>
            <div class="agent-desc">
              Ask any cooking or recipe question. This agent uses <strong>Retrieval-Augmented Generation (RAG)</strong>
              to search your uploaded recipe documents and answer with grounded, accurate information
              powered by IBM Granite.
            </div>
            <div class="mt-2">
              <span class="tag">RAG</span><span class="tag">Document Q&A</span>
              <span class="tag">IBM Granite</span><span class="tag">FAISS</span>
            </div>
          </div>
        </div>
        <div class="col-md-6">
          <div class="agent-card agent-2 h-100" onclick="showPanel('agent2')">
            <div class="agent-icon">🧑‍🍳</div>
            <div class="agent-title">Agent 2 – Personalized Recipe Generator</div>
            <div class="agent-desc">
              Generate fully personalised recipes based on your available ingredients, dietary needs,
              cuisine preference, and time constraints — powered by IBM Granite's instruction-following capability.
            </div>
            <div class="mt-2">
              <span class="tag">Recipe Gen</span><span class="tag">Personalised</span>
              <span class="tag">IBM Granite</span><span class="tag">Constraints</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Architecture diagram (SVG) -->
      <div class="card mb-4">
        <div class="card-header-ibm"><i class="bi bi-diagram-3 me-2"></i>System Architecture</div>
        <div class="card-body">
          <svg viewBox="0 0 760 200" xmlns="http://www.w3.org/2000/svg"
               style="width:100%;max-height:200px">
            <defs>
              <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#0f62fe"/>
              </marker>
            </defs>
            <!-- User -->
            <rect x="10" y="75" width="100" height="50" rx="8" fill="#edf5ff" stroke="#a6c8ff" stroke-width="1.5"/>
            <text x="60" y="96" text-anchor="middle" font-size="11" font-weight="600" fill="#0043ce">User</text>
            <text x="60" y="112" text-anchor="middle" font-size="9" fill="#555">Question /</text>
            <text x="60" y="124" text-anchor="middle" font-size="9" fill="#555">Preferences</text>
            <!-- Arrow -->
            <line x1="111" y1="100" x2="148" y2="100" stroke="#0f62fe" stroke-width="1.5" marker-end="url(#arr)"/>
            <!-- Flask -->
            <rect x="150" y="65" width="110" height="70" rx="8" fill="#fff" stroke="#d0e2ff" stroke-width="1.5"/>
            <text x="205" y="90" text-anchor="middle" font-size="11" font-weight="600" fill="#393939">Flask API</text>
            <text x="205" y="106" text-anchor="middle" font-size="9" fill="#555">Route Dispatcher</text>
            <text x="205" y="120" text-anchor="middle" font-size="9" fill="#555">Agent Orchestrator</text>
            <!-- Arrow up to Agent1 -->
            <line x1="260" y1="85" x2="310" y2="60" stroke="#0f62fe" stroke-width="1.5" marker-end="url(#arr)"/>
            <!-- Arrow down to Agent2 -->
            <line x1="260" y1="115" x2="310" y2="140" stroke="#6929c4" stroke-width="1.5" marker-end="url(#arr)"/>
            <!-- Agent 1 box -->
            <rect x="312" y="20" width="150" height="60" rx="8" fill="#edf5ff" stroke="#a6c8ff" stroke-width="1.5"/>
            <text x="387" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#0043ce">Agent 1</text>
            <text x="387" y="56" text-anchor="middle" font-size="9" fill="#555">Knowledge (RAG)</text>
            <text x="387" y="70" text-anchor="middle" font-size="9" fill="#555">FAISS + Granite</text>
            <!-- Agent 2 box -->
            <rect x="312" y="120" width="150" height="60" rx="8" fill="#f6f2ff" stroke="#d4bbff" stroke-width="1.5"/>
            <text x="387" y="142" text-anchor="middle" font-size="10" font-weight="700" fill="#6929c4">Agent 2</text>
            <text x="387" y="156" text-anchor="middle" font-size="9" fill="#555">Recipe Generator</text>
            <text x="387" y="170" text-anchor="middle" font-size="9" fill="#555">Granite Instruct</text>
            <!-- Arrows to watsonx -->
            <line x1="462" y1="50" x2="510" y2="70" stroke="#0f62fe" stroke-width="1.5" marker-end="url(#arr)"/>
            <line x1="462" y1="150" x2="510" y2="130" stroke="#6929c4" stroke-width="1.5" marker-end="url(#arr)"/>
            <!-- watsonx box -->
            <rect x="512" y="60" width="140" height="80" rx="8" fill="#001141" stroke="#4589ff" stroke-width="1.5"/>
            <text x="582" y="86" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">IBM watsonx.ai</text>
            <text x="582" y="102" text-anchor="middle" font-size="9" fill="#a6c8ff">Granite 13B Chat</text>
            <text x="582" y="116" text-anchor="middle" font-size="9" fill="#a6c8ff">Granite 13B Instruct</text>
            <text x="582" y="130" text-anchor="middle" font-size="9" fill="#78a9ff">Cloud API</text>
            <!-- FAISS arrow -->
            <rect x="380" y="0" width="80" height="16" rx="4" fill="#defbe6" stroke="#42be65" stroke-width="1"/>
            <text x="420" y="12" text-anchor="middle" font-size="8" fill="#198038" font-weight="600">FAISS VectorDB</text>
            <line x1="387" y1="20" x2="387" y2="16" stroke="#198038" stroke-width="1.2"/>
            <!-- Docs -->
            <rect x="652" y="75" width="95" height="50" rx="8" fill="#fff" stroke="#e0e0e0" stroke-width="1.5"/>
            <text x="700" y="96" text-anchor="middle" font-size="10" font-weight="600" fill="#393939">Uploaded</text>
            <text x="700" y="112" text-anchor="middle" font-size="10" fill="#393939">Documents</text>
            <line x1="652" y1="100" x2="655" y2="100" stroke="#198038" stroke-width="1.2"/>
            <line x1="420" y1="8" x2="680" y2="8" stroke="#198038" stroke-width="1" stroke-dasharray="4,3"/>
            <line x1="680" y1="8" x2="680" y2="75" stroke="#198038" stroke-width="1" stroke-dasharray="4,3"/>
          </svg>
        </div>
      </div>

      <!-- Quick start -->
      <div class="card">
        <div class="card-header-ibm"><i class="bi bi-lightning me-2"></i>Quick Start Guide</div>
        <div class="card-body">
          <ol class="mb-0" style="line-height:2">
            <li>Set <code>WATSONX_API_KEY</code> and <code>WATSONX_PROJECT_ID</code> in your <code>.env</code> file.</li>
            <li>Go to <strong>Manage Docs</strong> and upload your recipe PDF or text files to build the RAG knowledge base.</li>
            <li>Use <strong>Knowledge Agent</strong> to ask cooking questions grounded in your documents.</li>
            <li>Use <strong>Recipe Generator</strong> to create personalised recipes based on your preferences.</li>
          </ol>
        </div>
      </div>
    </div>

    <!-- ════════ PANEL: AGENT 1 ════════ -->
    <div id="panel-agent1" class="panel">
      <div class="section-title"><i class="bi bi-search-heart me-2"></i>Agent 1 – Recipe Knowledge Agent</div>
      <p class="text-muted small mb-3">
        Ask any cooking or recipe question. When documents are loaded, answers are grounded in your
        uploaded recipe files using <span class="rag-badge">RAG</span> + IBM Granite.
        Without documents, the agent uses Granite's general culinary knowledge.
      </p>

      <div class="card mb-3">
        <div class="card-header-ibm">
          <i class="bi bi-chat-dots me-2"></i>Recipe Q&amp;A Chat
          <span id="ragStatus" class="rag-badge ms-2 d-none">RAG Active</span>
        </div>
        <div class="card-body p-3">
          <div class="chat-wrap" id="chatWrap">
            <div class="bubble bubble-bot">
              👋 Hello! I'm your <strong>Recipe Knowledge Agent</strong> powered by IBM Granite.<br><br>
              Ask me anything about cooking, recipes, ingredients, or techniques.
              Upload recipe documents in <em>Manage Docs</em> to enable <strong>RAG</strong> — 
              I'll then answer based on your actual recipe files!<br><br>
              <em>Example questions:</em><br>
              • How do I make authentic vegetable biryani?<br>
              • What is the difference between baking soda and baking powder?<br>
              • What are common egg substitutes in baking?<br>
              • How do I make a sugar-free chocolate cake?
            </div>
          </div>
        </div>
        <div class="card-footer bg-white border-top p-2">
          <div class="d-flex gap-2">
            <textarea id="questionInput" class="form-control" rows="2"
              placeholder="Ask a cooking or recipe question…" style="resize:none"></textarea>
            <button class="btn btn-ibm px-3" id="askBtn" onclick="askAgent1()">
              <i class="bi bi-send-fill"></i>
            </button>
          </div>
          <div class="mt-2 d-flex flex-wrap gap-1" id="quickQuestions">
            <button class="btn btn-sm btn-outline-primary py-0"
              onclick="setQ('How do I make authentic vegetable biryani?')">Vegetable Biryani</button>
            <button class="btn btn-sm btn-outline-primary py-0"
              onclick="setQ('What is the difference between baking soda and baking powder?')">Baking Soda vs Powder</button>
            <button class="btn btn-sm btn-outline-primary py-0"
              onclick="setQ('What are common substitutes for eggs in baking?')">Egg Substitutes</button>
            <button class="btn btn-sm btn-outline-primary py-0"
              onclick="setQ('How do I make sugar-free chocolate cake?')">Sugar-Free Cake</button>
            <button class="btn btn-sm btn-outline-primary py-0"
              onclick="setQ('Which flour is best for gluten-free cakes?')">Gluten-Free Flour</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ PANEL: AGENT 2 ════════ -->
    <div id="panel-agent2" class="panel">
      <div class="section-title"><i class="bi bi-magic me-2"></i>Agent 2 – Personalized Recipe Generator</div>
      <p class="text-muted small mb-3">
        Fill in your preferences and constraints. IBM Granite will generate a complete,
        personalised recipe tailored exactly to your needs.
      </p>
      <div class="row g-3">
        <!-- Form -->
        <div class="col-lg-5">
          <div class="card h-100">
            <div class="card-header-granite">
              <i class="bi bi-sliders me-2"></i>Your Preferences
            </div>
            <div class="card-body">
              <div class="mb-2">
                <label class="form-label">Ingredients Available</label>
                <textarea class="form-control" id="pIngredients" rows="2"
                  placeholder="e.g. eggs, oats, milk, banana, almond flour…"></textarea>
              </div>
              <div class="row g-2 mb-2">
                <div class="col-6">
                  <label class="form-label">Meal Type</label>
                  <select class="form-select form-select-sm" id="pMealType">
                    <option>Breakfast</option><option>Lunch</option>
                    <option>Dinner</option><option>Snack</option>
                    <option>Dessert</option><option>Appetiser</option>
                  </select>
                </div>
                <div class="col-6">
                  <label class="form-label">Cuisine</label>
                  <select class="form-select form-select-sm" id="pCuisine">
                    <option>Any</option><option>Indian</option><option>Italian</option>
                    <option>Mexican</option><option>Chinese</option><option>Mediterranean</option>
                    <option>American</option><option>French</option><option>Japanese</option>
                    <option>Middle Eastern</option><option>Thai</option>
                  </select>
                </div>
              </div>
              <div class="row g-2 mb-2">
                <div class="col-6">
                  <label class="form-label">Cooking Time (mins)</label>
                  <select class="form-select form-select-sm" id="pCookTime">
                    <option value="15">Under 15 min</option>
                    <option value="30" selected>Under 30 min</option>
                    <option value="45">Under 45 min</option>
                    <option value="60">Under 1 hour</option>
                    <option value="90">Under 90 min</option>
                    <option value="120">2+ hours</option>
                  </select>
                </div>
                <div class="col-6">
                  <label class="form-label">Difficulty</label>
                  <select class="form-select form-select-sm" id="pDifficulty">
                    <option>Easy</option><option selected>Medium</option><option>Hard</option>
                  </select>
                </div>
              </div>
              <div class="row g-2 mb-2">
                <div class="col-6">
                  <label class="form-label">Servings</label>
                  <select class="form-select form-select-sm" id="pServings">
                    <option>1</option><option selected>2</option><option>4</option>
                    <option>6</option><option>8</option><option>10+</option>
                  </select>
                </div>
                <div class="col-6">
                  <label class="form-label">Dietary Restrictions</label>
                  <select class="form-select form-select-sm" id="pDietary">
                    <option>None</option><option>Vegetarian</option><option>Vegan</option>
                    <option>Gluten-Free</option><option>Keto</option><option>Paleo</option>
                    <option>Dairy-Free</option><option>Low-Carb</option>
                    <option>Low-Calorie</option><option>High-Protein</option>
                  </select>
                </div>
              </div>
              <div class="mb-2">
                <label class="form-label">Allergies</label>
                <input class="form-control form-control-sm" id="pAllergies"
                  placeholder="e.g. peanuts, shellfish, tree nuts…"/>
              </div>
              <div class="mb-3">
                <label class="form-label">Special Notes / Additional Requests</label>
                <textarea class="form-control" id="pNotes" rows="2"
                  placeholder="e.g. high-protein, no onions, baby-friendly, low sodium…"></textarea>
              </div>

              <!-- Quick presets -->
              <div class="mb-3">
                <div class="form-label">Quick Presets</div>
                <div class="d-flex flex-wrap gap-1">
                  <button class="btn btn-sm btn-outline-secondary py-0"
                    onclick="loadPreset('protein_breakfast')">💪 High-Protein Breakfast</button>
                  <button class="btn btn-sm btn-outline-secondary py-0"
                    onclick="loadPreset('vegan_pasta')">🌱 Vegan Pasta</button>
                  <button class="btn btn-sm btn-outline-secondary py-0"
                    onclick="loadPreset('keto_dessert')">🍫 Keto Dessert</button>
                  <button class="btn btn-sm btn-outline-secondary py-0"
                    onclick="loadPreset('indian_dinner')">🍛 Indian Dinner</button>
                </div>
              </div>

              <button class="btn btn-granite w-100" onclick="generateRecipe()" id="genBtn">
                <i class="bi bi-magic me-2"></i>Generate My Recipe
              </button>
            </div>
          </div>
        </div>
        <!-- Output -->
        <div class="col-lg-7">
          <div class="card h-100">
            <div class="card-header-granite">
              <i class="bi bi-file-earmark-richtext me-2"></i>Generated Recipe
              <button class="btn btn-sm btn-light float-end py-0" id="copyRecipeBtn"
                onclick="copyRecipe()" style="display:none">
                <i class="bi bi-clipboard me-1"></i>Copy
              </button>
            </div>
            <div class="card-body">
              <div id="recipeOutput" class="recipe-output">
                <div class="text-center text-muted py-5">
                  <i class="bi bi-file-earmark-richtext display-4 d-block mb-2 opacity-25"></i>
                  Your personalised recipe will appear here.<br>
                  <small>Fill in your preferences and click <strong>Generate My Recipe</strong>.</small>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ PANEL: DOCS ════════ -->
    <div id="panel-docs" class="panel">
      <div class="section-title"><i class="bi bi-file-earmark-text me-2"></i>Manage Recipe Documents</div>
      <p class="text-muted small mb-3">
        Upload PDF or text files containing recipes. These will be chunked, embedded, and stored in
        a <strong>FAISS</strong> vector database to power the RAG pipeline in Agent 1.
      </p>
      <div class="row g-3">
        <div class="col-md-5">
          <div class="card">
            <div class="card-header-ibm"><i class="bi bi-cloud-upload me-2"></i>Upload Document</div>
            <div class="card-body">
              <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()"
                ondragover="event.preventDefault();this.classList.add('drag')"
                ondragleave="this.classList.remove('drag')"
                ondrop="handleDrop(event)">
                <i class="bi bi-file-earmark-arrow-up d-block"></i>
                <div class="fw-semibold mb-1">Drop file here or click to browse</div>
                <div class="text-muted small">Supported: PDF, TXT, MD • Max 16 MB</div>
              </div>
              <input type="file" id="fileInput" accept=".pdf,.txt,.md" class="d-none"
                onchange="uploadFile(this.files[0])"/>
              <div id="uploadStatus" class="mt-2"></div>
            </div>
          </div>
        </div>
        <div class="col-md-7">
          <div class="card">
            <div class="card-header-ibm">
              <i class="bi bi-database me-2"></i>Knowledge Base
              <span class="badge bg-light text-dark ms-2" id="docCountBadge">0 docs</span>
            </div>
            <div class="card-body">
              <div id="docList">
                <div class="text-center text-muted py-3 small">No documents uploaded yet.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ PANEL: CONFIG ════════ -->
    <div id="panel-config" class="panel">
      <div class="section-title"><i class="bi bi-gear me-2"></i>Configuration &amp; System Status</div>
      <div class="row g-3">
        <div class="col-md-6">
          <div class="card">
            <div class="card-header-ibm"><i class="bi bi-key me-2"></i>IBM watsonx.ai Credentials</div>
            <div class="card-body">
              <p class="small text-muted">
                Set these in a <code>.env</code> file alongside <code>app.py</code>.
                Never hard-code credentials in source files.
              </p>
              <div class="mb-3">
                <label class="form-label"><code>WATSONX_API_KEY</code></label>
                <input type="password" class="form-control form-control-sm" id="cfgApiKey"
                  placeholder="Paste your IBM Cloud API key…"/>
              </div>
              <div class="mb-3">
                <label class="form-label"><code>WATSONX_PROJECT_ID</code></label>
                <input type="text" class="form-control form-control-sm" id="cfgProjectId"
                  placeholder="Your watsonx.ai project ID…"/>
              </div>
              <div class="mb-3">
                <label class="form-label"><code>WATSONX_URL</code> (region endpoint)</label>
                <select class="form-select form-select-sm" id="cfgUrl">
                  <option value="https://us-south.ml.cloud.ibm.com">US South (Dallas)</option>
                  <option value="https://eu-de.ml.cloud.ibm.com">EU (Frankfurt)</option>
                  <option value="https://eu-gb.ml.cloud.ibm.com">EU (London)</option>
                  <option value="https://jp-tok.ml.cloud.ibm.com">AP (Tokyo)</option>
                  <option value="https://au-syd.ml.cloud.ibm.com">AP (Sydney)</option>
                </select>
              </div>
              <button class="btn btn-ibm btn-sm" onclick="saveConfig()">
                <i class="bi bi-save me-1"></i>Save to .env (manual)
              </button>
              <div id="cfgMsg" class="mt-2"></div>
            </div>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card">
            <div class="card-header-ibm"><i class="bi bi-activity me-2"></i>System Status</div>
            <div class="card-body" id="sysStatus">
              <div class="text-center text-muted py-3">Loading…</div>
            </div>
          </div>
          <div class="card mt-3">
            <div class="card-header-ibm"><i class="bi bi-book me-2"></i>IBM Granite Models Used</div>
            <div class="card-body small">
              <table class="table table-sm table-borderless mb-0">
                <tr>
                  <td class="fw-semibold">Agent 1 (RAG Q&amp;A)</td>
                  <td><code>ibm/granite-13b-chat-v2</code></td>
                </tr>
                <tr>
                  <td class="fw-semibold">Agent 2 (Recipe Gen)</td>
                  <td><code>ibm/granite-13b-instruct-v2</code></td>
                </tr>
                <tr>
                  <td class="fw-semibold">Embeddings</td>
                  <td><code>all-MiniLM-L6-v2</code></td>
                </tr>
                <tr>
                  <td class="fw-semibold">Vector DB</td>
                  <td><code>FAISS (in-memory)</code></td>
                </tr>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /main -->
</div><!-- /body layout -->

<div class="footer-bar">
  RecipeGenie AI &nbsp;|&nbsp; Powered by <strong>IBM watsonx.ai</strong> Granite Models &nbsp;|&nbsp;
  RAG &bull; FAISS &bull; LangChain &bull; Flask &nbsp;|&nbsp;
  <span style="opacity:.6">Multi-Agent AI Demo</span>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ── Panel navigation ─────────────────────────────────────────────────────────
function showPanel(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  document.querySelectorAll('.sidebar .nav-link').forEach(l => l.classList.remove('active'));
  const link = document.querySelector(`.sidebar .nav-link[onclick*="${id}"]`);
  if (link) link.classList.add('active');
  window.scrollTo(0, 0);
}

// ── Status check ─────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch('/api/config-status');
    const d = await r.json();
    const dot  = document.getElementById('statusDot');
    const txt  = document.getElementById('statusText');
    const rag  = document.getElementById('ragStatus');
    if (d.configured) {
      dot.className = 'status-dot dot-ok';
      txt.textContent = 'watsonx.ai Connected';
    } else {
      dot.className = 'status-dot dot-warn';
      txt.textContent = 'Credentials needed';
    }
    if (d.vector_store_ready && rag) rag.classList.remove('d-none');
    else if (rag) rag.classList.add('d-none');

    document.getElementById('docCountBadge').textContent = d.docs_loaded + ' docs';
    renderSysStatus(d);
  } catch(e) {
    document.getElementById('statusDot').className = 'status-dot dot-err';
    document.getElementById('statusText').textContent = 'Server error';
  }
}

function renderSysStatus(d) {
  const el = document.getElementById('sysStatus');
  if (!el) return;
  el.innerHTML = `
    <div class="d-flex flex-column gap-2">
      ${statusRow('IBM watsonx.ai Credentials', d.configured)}
      ${statusRow('FAISS Vector Store', d.vector_store_ready)}
      ${statusRow('Documents Loaded', d.docs_loaded > 0, d.docs_loaded + ' files')}
      <div class="mt-2 small text-muted">
        <div>Chat model: <code>${d.model_chat}</code></div>
        <div>Instruct model: <code>${d.model_inst}</code></div>
      </div>
    </div>`;
}
function statusRow(label, ok, extra='') {
  const icon  = ok ? '✅' : '⚠️';
  const color = ok ? 'text-success' : 'text-warning';
  return `<div class="d-flex justify-content-between align-items-center border-bottom pb-1">
    <span class="small">${label}</span>
    <span class="small ${color} fw-semibold">${icon} ${extra || (ok?'Ready':'Not set')}</span>
  </div>`;
}

// ── Agent 1 – Chat ────────────────────────────────────────────────────────────
function setQ(q) {
  document.getElementById('questionInput').value = q;
  document.getElementById('questionInput').focus();
}

async function askAgent1() {
  const input = document.getElementById('questionInput');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';

  appendBubble('user', q);
  const thinkId = appendThinking();
  document.getElementById('askBtn').disabled = true;

  try {
    const r  = await fetch('/api/agent1/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const d = await r.json();
    removeThinking(thinkId);
    if (d.error) {
      appendBubble('bot', '⚠️ ' + d.error, true);
    } else {
      const meta = `🤖 IBM Granite  |  ${d.source}  |  Chunks: ${d.chunks_used}`;
      appendBubble('bot', d.answer, false, meta);
    }
  } catch(e) {
    removeThinking(thinkId);
    appendBubble('bot', '⚠️ Network error: ' + e.message, true);
  }
  document.getElementById('askBtn').disabled = false;
  checkStatus();
}

function appendBubble(role, text, isError=false, meta='') {
  const wrap = document.getElementById('chatWrap');
  const div  = document.createElement('div');
  let cls = role === 'user' ? 'bubble bubble-user' : 'bubble bubble-bot';
  if (isError) cls += ' bubble-error';
  div.className = cls;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'bubble-meta'; m.textContent = meta;
    div.appendChild(m);
  }
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}

function appendThinking() {
  const wrap = document.getElementById('chatWrap');
  const id   = 'think_' + Date.now();
  wrap.innerHTML += `<div id="${id}" class="thinking">
    <div class="dots"><span></span><span></span><span></span></div>
    IBM Granite is thinking…
  </div>`;
  wrap.scrollTop = wrap.scrollHeight;
  return id;
}
function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Agent 2 – Recipe generator ────────────────────────────────────────────────
const PRESETS = {
  protein_breakfast: {
    ingredients:'eggs, oats, Greek yogurt, banana, whey protein, almond milk',
    meal_type:'Breakfast', cuisine:'Any', cooking_time:'15',
    difficulty:'Easy', servings:'2', dietary:'High-Protein',
    allergies:'none', notes:'High-protein, filling, quick morning meal'
  },
  vegan_pasta: {
    ingredients:'pasta, cherry tomatoes, spinach, garlic, olive oil, nutritional yeast, basil',
    meal_type:'Dinner', cuisine:'Italian', cooking_time:'20',
    difficulty:'Easy', servings:'2', dietary:'Vegan',
    allergies:'none', notes:'Quick weeknight vegan dinner'
  },
  keto_dessert: {
    ingredients:'almond flour, cocoa powder, eggs, butter, erythritol, vanilla extract',
    meal_type:'Dessert', cuisine:'Any', cooking_time:'30',
    difficulty:'Medium', servings:'8', dietary:'Keto',
    allergies:'none', notes:'Low-carb, sugar-free, rich chocolate dessert'
  },
  indian_dinner: {
    ingredients:'paneer, tomatoes, cream, garam masala, ginger, garlic, turmeric, cumin',
    meal_type:'Dinner', cuisine:'Indian', cooking_time:'45',
    difficulty:'Medium', servings:'4', dietary:'Vegetarian',
    allergies:'none', notes:'No onions, restaurant-style'
  }
};

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById('pIngredients').value = p.ingredients;
  document.getElementById('pMealType').value    = p.meal_type;
  document.getElementById('pCuisine').value     = p.cuisine;
  document.getElementById('pCookTime').value    = p.cooking_time;
  document.getElementById('pDifficulty').value  = p.difficulty;
  document.getElementById('pServings').value    = p.servings;
  document.getElementById('pDietary').value     = p.dietary;
  document.getElementById('pAllergies').value   = p.allergies;
  document.getElementById('pNotes').value       = p.notes;
}

async function generateRecipe() {
  const btn = document.getElementById('genBtn');
  const out = document.getElementById('recipeOutput');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating…';
  out.innerHTML = `<div class="thinking"><div class="dots"><span></span><span></span><span></span></div>
    IBM Granite is crafting your personalised recipe…</div>`;

  const payload = {
    ingredients  : document.getElementById('pIngredients').value || 'pantry staples',
    meal_type    : document.getElementById('pMealType').value,
    cuisine      : document.getElementById('pCuisine').value,
    cooking_time : document.getElementById('pCookTime').value,
    difficulty   : document.getElementById('pDifficulty').value,
    servings     : document.getElementById('pServings').value,
    dietary      : document.getElementById('pDietary').value,
    allergies    : document.getElementById('pAllergies').value || 'none',
    notes        : document.getElementById('pNotes').value || 'none',
  };

  try {
    const r = await fetch('/api/agent2/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (d.error) {
      out.innerHTML = `<div class="text-danger"><strong>⚠️ Error:</strong> ${d.error}</div>`;
    } else {
      out.textContent = d.recipe;
      document.getElementById('copyRecipeBtn').style.display = '';
    }
  } catch(e) {
    out.innerHTML = `<div class="text-danger">⚠️ Network error: ${e.message}</div>`;
  }
  btn.disabled = false;
  btn.innerHTML = '<i class="bi bi-magic me-2"></i>Generate My Recipe';
}

function copyRecipe() {
  const text = document.getElementById('recipeOutput').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyRecipeBtn');
    btn.innerHTML = '<i class="bi bi-check me-1"></i>Copied!';
    setTimeout(() => btn.innerHTML = '<i class="bi bi-clipboard me-1"></i>Copy', 2000);
  });
}

// ── Document management ────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (!file) return;
  const status = document.getElementById('uploadStatus');
  status.innerHTML = `<div class="text-muted small">
    <span class="spinner-border spinner-border-sm me-1"></span>Uploading ${file.name}…</div>`;

  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload', {method:'POST', body: fd});
    const d = await r.json();
    if (d.error) {
      status.innerHTML = `<div class="text-danger small">⚠️ ${d.error}</div>`;
    } else {
      status.innerHTML = `<div class="text-success small">
        ✅ <strong>${d.name}</strong> uploaded — ${d.chars.toLocaleString()} chars indexed.</div>`;
      loadDocs();
      checkStatus();
    }
  } catch(e) {
    status.innerHTML = `<div class="text-danger small">⚠️ ${e.message}</div>`;
  }
  document.getElementById('fileInput').value = '';
}

async function loadDocs() {
  try {
    const r = await fetch('/api/documents');
    const d = await r.json();
    const list = document.getElementById('docList');
    document.getElementById('docCountBadge').textContent = d.count + ' docs';
    if (d.count === 0) {
      list.innerHTML = '<div class="text-center text-muted py-3 small">No documents uploaded yet.</div>';
      return;
    }
    list.innerHTML = d.documents.map(doc => `
      <div class="doc-item">
        <div>
          <div class="doc-name"><i class="bi bi-file-earmark-text me-1 text-primary"></i>${doc.name}</div>
          <div class="doc-meta">${doc.chars.toLocaleString()} chars &bull; ${doc.uploaded.slice(0,16).replace('T',' ')}</div>
        </div>
        <button class="btn btn-sm btn-outline-danger py-0 px-2"
          onclick="deleteDoc('${doc.id}','${doc.name}')">
          <i class="bi bi-trash3"></i>
        </button>
      </div>`).join('');
  } catch(e) { console.error(e); }
}

async function deleteDoc(id, name) {
  if (!confirm(`Remove "${name}" from the knowledge base?`)) return;
  const r = await fetch('/api/documents/' + id, {method:'DELETE'});
  const d = await r.json();
  if (d.success) { loadDocs(); checkStatus(); }
}

// ── Config panel ──────────────────────────────────────────────────────────────
function saveConfig() {
  const msg = document.getElementById('cfgMsg');
  msg.innerHTML = `<div class="alert alert-info py-1 small mt-2 mb-0">
    ℹ️ Add these lines to your <code>.env</code> file and restart the server:<br>
    <code>WATSONX_API_KEY=${document.getElementById('cfgApiKey').value}</code><br>
    <code>WATSONX_PROJECT_ID=${document.getElementById('cfgProjectId').value}</code><br>
    <code>WATSONX_URL=${document.getElementById('cfgUrl').value}</code>
  </div>`;
}

// ── Enter key for chat ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('questionInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askAgent1(); }
  });
  checkStatus();
  loadDocs();
  setInterval(checkStatus, 30000);
});
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Force UTF-8 output on Windows so emoji characters print cleanly
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("  RecipeGenie AI - IBM watsonx.ai Granite Demo")
    print("=" * 60)
    if not WATSONX_API_KEY:
        print("  [!] WATSONX_API_KEY not set - add to .env file")
    if not WATSONX_PROJECT_ID:
        print("  [!] WATSONX_PROJECT_ID not set - add to .env file")
    if WATSONX_API_KEY and WATSONX_PROJECT_ID:
        print("  [OK] IBM watsonx.ai credentials detected")
    print("  [*] Starting server at http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
