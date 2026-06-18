# app.py – Fixed focus, case‑insensitive document filtering
import os
import uuid
import re
from flask import Flask, request, jsonify, render_template, session
from brain import KnowledgeBrain
from memory import ConversationMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("🧠 Loading Knowledge Brain...")
brain = KnowledgeBrain(
    pdf_directory="./pdfs",
    persist_directory="./vector_store"
)

api_key = os.getenv("GROQ_API_KEY", "your_api_key")
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

memory = ConversationMemory()
session_focus = {}          # session_id → exact filename or None

# Improved prompt – forces use of context
PROMPT = """You are a helpful assistant. Answer using ONLY the context below.
If the context contains relevant information, you MUST use it to answer.
Do NOT say "I don't have that information" if the answer is in the context.
Be direct and thorough. If the user asks for a detailed answer, provide one.

Active document filter: {focus_info}

Context from documents:
{context}

Conversation history:
{chat_history}

Question: {question}
Answer:"""

QA_PROMPT = ChatPromptTemplate.from_template(PROMPT)

def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get('source_file', '?')
        if src in seen:
            continue
        seen.add(src)
        page = doc.metadata.get('page_number', '?')
        parts.append(f"[{src} p{page}]\n{doc.page_content[:800]}\n")
    return "\n".join(parts)

# ------------------------------------------------------------
# Focus management – case‑insensitive, exact filename resolution
# ------------------------------------------------------------
def resolve_filename(user_input: str) -> str | None:
    """Try to match user input to an existing document filename (case‑insensitive)."""
    all_files = brain.get_all_filenames()
    # direct match (case‑insensitive)
    for f in all_files:
        if user_input.lower() == f.lower():
            return f
    # if user omitted .pdf, try adding it
    if not user_input.lower().endswith('.pdf'):
        for f in all_files:
            if f.lower() == user_input.lower() + '.pdf':
                return f
    return None

def detect_focus_command(question: str) -> str | None:
    """
    Returns:
        - exact filename (from brain) if a valid focus command is found
        - "CLEAR" if user wants to remove focus
        - None otherwise
    """
    q = question.lower()
    # patterns for "only use X.pdf" / "focus on X.pdf"
    patterns = [
        r'only\s+use\s+([\w\-.]+(?:\.pdf)?)',
        r'focus\s+on\s+([\w\-.]+(?:\.pdf)?)',
        r'use\s+only\s+([\w\-.]+(?:\.pdf)?)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            candidate = m.group(1).strip()
            # resolve to real filename
            resolved = resolve_filename(candidate)
            if resolved:
                return resolved
            else:
                # no matching file – we still return the candidate so the user can be notified
                return candidate  # will be handled as "file not found" in the route
    if "clear focus" in q or "remove focus" in q or "no filter" in q:
        return "CLEAR"
    return None

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", session.get("session_id", "default"))
    category = data.get("category", "all")

    if not question:
        return jsonify({"error": "Question required"}), 400

    # --- Focus commands ------------------------------------------------
    focus_file = session_focus.get(session_id)
    focus_cmd = detect_focus_command(question)

    if focus_cmd == "CLEAR":
        session_focus.pop(session_id, None)
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", "✅ Document filter cleared. Searching all documents again.")
        return jsonify({"answer": "✅ Document filter cleared. Searching all documents again.", "sources": [], "focus": None})
    elif focus_cmd is not None:
        # check if the focus_cmd is a real filename or just a candidate that doesn't exist
        all_files = brain.get_all_filenames()
        if focus_cmd in all_files:
            session_focus[session_id] = focus_cmd
            msg = f"✅ Now focusing on **{focus_cmd}** only. All subsequent questions will be answered from this document."
        else:
            # file not found
            msg = f"❌ Document `{focus_cmd}` not found in the knowledge base. Available: {', '.join(all_files)}"
        memory.add_message(session_id, "user", question)
        memory.add_message(session_id, "assistant", msg)
        return jsonify({"answer": msg, "sources": [], "focus": session_focus.get(session_id)})
    # --------------------------------------------------------------------

    # Get history
    history = memory.get_history(session_id, last_n=6)
    history_str = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    ) if history else "No history"

    # --- Retrieval with focus -------------------------------------------
    if focus_file:
        # Get many candidates and filter strictly
        raw_docs = brain.search(question, k=30, category=category if category != "all" else None)
        # case‑insensitive filter
        docs = [d for d in raw_docs if d.metadata.get('source_file', '').lower() == focus_file.lower()]
        if not docs:
            # No matches inside focused document – tell the user
            memory.add_message(session_id, "user", question)
            answer = f"I could not find any relevant information inside **{focus_file}**."
            memory.add_message(session_id, "assistant", answer)
            return jsonify({"answer": answer, "sources": [], "focus": focus_file})
        # take top 6 chunks (more context for better answers)
        docs = docs[:6]
    else:
        docs = brain.intelligent_search(question, k=4, category=category if category != "all" else None)
    # --------------------------------------------------------------------

    context = format_docs(docs)

    focus_info = f"Currently focused on: {focus_file}. Only use this document." if focus_file else "No active document filter."

    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history_str,
        "question": question,
        "focus_info": focus_info
    })

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer)

    # Sources (deduplicated)
    seen_src = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source_file", "?")
        if src not in seen_src:
            seen_src.add(src)
            sources.append({"document": src, "page": doc.metadata.get("page_number", "?")})

    return jsonify({"answer": answer, "sources": sources, "focus": focus_file})

@app.route("/api/focus", methods=["POST"])
def set_focus():
    data = request.get_json()
    session_id = data.get("session_id", session.get("session_id", "default"))
    filename = data.get("filename", None)
    if filename:
        session_focus[session_id] = filename
    else:
        session_focus.pop(session_id, None)
    return jsonify({"focus": session_focus.get(session_id)})

@app.route("/api/stats")
def stats():
    return jsonify(brain.get_stats())

@app.route("/api/documents")
def documents():
    docs = []
    for fname, meta in brain.documents_metadata.items():
        docs.append({
            "filename": fname,
            "pages": meta.get("pages", 0),
            "chunks": meta.get("chunks", 0),
            "category": meta.get("category", "general")
        })
    return jsonify({"documents": docs, "total": len(docs)})

@app.route("/api/categories")
def categories():
    return jsonify({"categories": brain.get_categories()})

@app.route("/api/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    memory.clear_session(session_id)
    session_focus.pop(session_id, None)
    return jsonify({"success": True})

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Chatbot Ready: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)