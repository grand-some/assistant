"""
Streamlit + OpenAI Assistants API demo
- Refactors the previous LangChain agent into an OpenAI Assistant that can call local tools
- Shows conversation history in the UI
- Lets the user supply their own OpenAI API key via the sidebar
- Adds a GitHub repo link in the sidebar

How to run
----------
1) Create and activate a virtualenv
2) pip install -r requirements.txt  (see minimal list below)
3) streamlit run streamlit_assistants_app.py

Minimal requirements (put these lines into requirements.txt):
streamlit>=1.36
openai>=1.40.0
langchain>=0.2.10
duckduckgo-search>=6.1.9
wikipedia>=1.4.0
requests>=2.31.0
beautifulsoup4>=4.12

Notes
-----
- This app demonstrates the "function tool calls" pattern from the two reference codes.
- It wires DuckDuckGo, Wikipedia, Web scraping, and Save-to-TXT as callable tools.
- The Assistant is created on first use and cached in st.session_state.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import streamlit as st

# OpenAI new-style SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Optional libs used by tools
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import wikipedia

# ----------------------------
# Sidebar: API key + GitHub link
# ----------------------------
st.set_page_config(page_title="Research Assistant (OpenAI Assistants)", page_icon="🔎", layout="wide")

with st.sidebar:
    st.header("🔧 Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your key is used only in your browser session.")
    st.divider()
    st.header("📦 Project Repo")
    st.markdown(
        "[View code on GitHub](https://github.com/grand-some/assistant.git)",
        help="Replace this link with your actual repository URL."
    )
    st.caption("This demo refactors the previous agent into an OpenAI Assistant with local tools.")

st.title("🔎 Research Assistant — OpenAI Assistants + Streamlit")
st.write(
    "Ask questions that require web search, Wikipedia lookups, and page scraping. "
    "The assistant will call local tools as needed and save a consolidated .txt file."
)

# ----------------------------
# Session state init
# ----------------------------
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # local display cache (role, content)

# ----------------------------
# Utilities: Local tools (mirroring first code)
# ----------------------------

def tool_duckduckgo_search(query: str) -> str:
    """Search web using DuckDuckGo and return a compact textual summary of top results.
    We return a JSON string with a list of {title, href, snippet}.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=8):
            results.append({
                "title": r.get("title"),
                "href": r.get("href"),
                "snippet": r.get("body"),
            })
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


def tool_wikipedia_search(query: str) -> str:
    """Search wikipedia and return the summary of the top page.
    Returns JSON with {title, summary, url}.
    """
    try:
        page_title = wikipedia.search(query, results=1)
        if not page_title:
            return json.dumps({"error": "No wikipedia results"}, ensure_ascii=False)
        page_title = page_title[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        return json.dumps({
            "title": page.title,
            "summary": wikipedia.summary(page.title, sentences=5, auto_suggest=False),
            "url": page.url,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def tool_web_scrape(url: str) -> str:
    """Fetch a page and return extracted main text.
    Returns JSON with {url, text}.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # crude text extraction
        texts = [t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "h1", "h2", "h3"])][:200]
        return json.dumps({"url": url, "text": "\n\n".join(texts)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def tool_save_to_txt(text: str) -> str:
    """Save compiled research to research_results.txt and return a message."""
    try:
        with open("research_results.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return "Research results saved to research_results.txt"
    except Exception as e:
        return f"Failed to save: {e}"


# Mapping name -> callable
FUNCTIONS_MAP = {
    "duckduckgo_search": tool_duckduckgo_search,
    "wikipedia_search": tool_wikipedia_search,
    "web_scrape": tool_web_scrape,
    "save_to_txt": tool_save_to_txt,
}

# JSON schema presented to the Assistant for callable functions
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search the web via DuckDuckGo; return top results as JSON (title, href, snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia and return a JSON with title, summary, url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to find on Wikipedia"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": "Fetch a URL and return main textual content as JSON with url+text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_txt",
            "description": "Save the compiled research text to a local research_results.txt file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to save"}
                },
                "required": ["text"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a research expert. Use Wikipedia and DuckDuckGo to gather comprehensive, accurate info. "
    "When DuckDuckGo yields relevant sites, call web_scrape to read them. Combine sources and produce a detailed, "
    "well-cited answer with clickable URLs. Finally, call save_to_txt with the compiled research."
)

# ----------------------------
# OpenAI helpers
# ----------------------------

def ensure_openai_client(key: str) -> OpenAI:
    if not key:
        raise ValueError("Please enter your OpenAI API key in the sidebar.")
    if OpenAI is None:
        raise RuntimeError("openai SDK is not installed. Please `pip install openai`.")
    return OpenAI(api_key=key)


def ensure_assistant(client: OpenAI) -> str:
    """Create the Assistant once and keep its id in session_state."""
    if st.session_state.assistant_id:
        return st.session_state.assistant_id
    # Create an Assistant configured with our function tools
    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions=SYSTEM_PROMPT,
        model="gpt-4o-mini",
        tools=TOOLS_SPEC,
    )
    st.session_state.assistant_id = assistant.id
    return assistant.id


def ensure_thread(client: OpenAI) -> str:
    if st.session_state.thread_id:
        return st.session_state.thread_id
    thread = client.beta.threads.create(messages=[])
    st.session_state.thread_id = thread.id
    return thread.id


def add_user_message(client: OpenAI, thread_id: str, content: str) -> None:
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=content)


def list_messages(client: OpenAI, thread_id: str) -> List[Dict[str, str]]:
    msgs = client.beta.threads.messages.list(thread_id=thread_id)
    out: List[Dict[str, str]] = []
    # messages.list returns newest-first; flip to oldest-first for display
    items = list(msgs)
    items.reverse()
    for m in items:
        role = m.role
        # content may include images, for simplicity we only show text entries
        if m.content and getattr(m.content[0], "text", None):
            out.append({"role": role, "content": m.content[0].text.value})
    return out


def run_assistant(client: OpenAI, thread_id: str, assistant_id: str):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    poll_until_complete_or_action(client, thread_id, run.id)


def poll_until_complete_or_action(client: OpenAI, thread_id: str, run_id: str):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        status = run.status
        if status in ("queued", "in_progress"):
            time.sleep(0.8)
            continue
        if status == "requires_action":
            # execute tool calls
            handle_tool_calls(client, thread_id, run)
            # after submitting tool outputs, loop again
            time.sleep(0.4)
            continue
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(0.5)


def handle_tool_calls(client: OpenAI, thread_id: str, run) -> None:
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    outputs = []
    for call in tool_calls:
        fname = call.function.name
        fargs = json.loads(call.function.arguments or "{}")
        st.status("Running tool: %s" % fname)
        try:
            result = FUNCTIONS_MAP[fname](**fargs)
        except Exception as e:
            result = json.dumps({"error": str(e)})
        outputs.append({"tool_call_id": call.id, "output": result})
    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=outputs,
    )

# ----------------------------
# Main App UI
# ----------------------------

with st.form("chat_input", clear_on_submit=True):
    user_query = st.text_input("Ask me anything (e.g., Research about the XZ backdoor)")
    submitted = st.form_submit_button("Send", type="primary")

if submitted:
    try:
        client = ensure_openai_client(api_key)
        assistant_id = ensure_assistant(client)
        thread_id = ensure_thread(client)
        add_user_message(client, thread_id, user_query)
        run_assistant(client, thread_id, assistant_id)
        # refresh local display cache
        st.session_state.messages = list_messages(client, thread_id)
    except Exception as e:
        st.error(str(e))

# Conversation history panel
st.subheader("Conversation")

if api_key and st.session_state.thread_id:
    # If we have a thread, fetch and show messages live
    try:
        client = ensure_openai_client(api_key)
        st.session_state.messages = list_messages(client, st.session_state.thread_id)
    except Exception as e:
        st.warning(f"Could not refresh messages: {e}")

for m in st.session_state.messages:
    role = "assistant" if m["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.write(m["content"])
