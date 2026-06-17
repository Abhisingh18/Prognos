"""
Q3 — AI AGENT: Excel topics -> Blog posts -> Google Docs -> update sheet
============================================================================
Reads topics from an Excel sheet, writes a blog per topic, saves each blog
to a Google Doc, and writes the Doc link back into the sheet.

This is a runnable SKELETON. The LLM call, Google Docs call, and Sheets I/O
are isolated behind small functions so you can plug in real credentials.

ASSUMPTIONS
-----------
1. Excel columns: Category | Topic | Updated Date | Blog link
   We only process rows where "Blog link" is empty (idempotent re-runs).
2. Google Docs + Drive + Sheets access via a service account
   (env: GOOGLE_APPLICATION_CREDENTIALS). Stubbed here.
3. One blog per topic. Failures on one row do NOT stop the others.
4. LLM is Anthropic Claude (swap freely).

GRAPH DESIGN (nodes / edges / state)
------------------------------------
STATE (shared dict passed between nodes):
    rows          : list of pending {category, topic, row_index} to process
    current       : the row currently being worked on
    draft         : generated blog text for `current`
    doc_url       : Google Doc URL for `current`
    completed     : list of finished rows
    errors        : list of (row_index, error)

NODES:
    load_topics   : read Excel -> fill state["rows"]
    pick_next     : pop one pending row -> state["current"] (router source)
    write_blog    : LLM generates blog for state["current"] -> state["draft"]
    save_to_doc   : create Google Doc, write draft -> state["doc_url"]
    update_sheet  : write doc_url back to that row's "Blog link" cell
    finalize      : save the workbook, report summary

EDGES:
    START -> load_topics -> pick_next
    pick_next --(conditional)--> write_blog       if a row remains
    pick_next --(conditional)--> finalize         if none remain
    write_blog -> save_to_doc -> update_sheet -> pick_next   (the LOOP)
    finalize -> END

The loop is the cycle pick_next -> write_blog -> save_to_doc ->
update_sheet -> pick_next, which LangGraph supports natively via a
conditional edge that routes back to pick_next until the queue is empty.
"""

from typing import TypedDict, List, Dict, Optional
# from langgraph.graph import StateGraph, START, END   # real import
# import openpyxl
# from anthropic import Anthropic
# from googleapiclient.discovery import build


# ---------- STATE ----------
class AgentState(TypedDict):
    rows: List[Dict]
    current: Optional[Dict]
    draft: Optional[str]
    doc_url: Optional[str]
    completed: List[Dict]
    errors: List[Dict]
    xlsx_path: str


# ---------- TOOL STUBS (replace with real APIs) ----------
def read_excel_rows(path: str) -> List[Dict]:
    """Return rows missing a blog link. Replace with openpyxl."""
    # wb = openpyxl.load_workbook(path); ws = wb.active
    # rows = []
    # for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
    #     category, topic, updated, blog_link = row
    #     if topic and not blog_link:
    #         rows.append({"row_index": i, "category": category, "topic": topic})
    # return rows
    return [
        {"row_index": 2, "category": "Marketing",
         "topic": "AI in marketing - top 5 Tools"},
        {"row_index": 3, "category": "Technology",
         "topic": "How MCP is changing the AI landscape"},
    ]


def llm_write_blog(category: str, topic: str) -> str:
    """Generate a blog post. Replace with a real Claude call."""
    # client = Anthropic()
    # msg = client.messages.create(
    #     model="claude-sonnet-4-6", max_tokens=2000,
    #     messages=[{"role": "user", "content":
    #         f"Write a ~700-word blog post. Category: {category}. "
    #         f"Topic: {topic}. Include a title, intro, 3-4 sections, conclusion."}])
    # return msg.content[0].text
    return f"# {topic}\n\n[{category}] Full ~700 word blog body here..."


def create_google_doc(title: str, body: str) -> str:
    """Create a Google Doc, insert body, return its URL. Replace with Docs API."""
    # docs = build("docs", "v1"); drive = build("drive", "v1")
    # doc = docs.documents().create(body={"title": title}).execute()
    # doc_id = doc["documentId"]
    # docs.documents().batchUpdate(documentId=doc_id, body={"requests":[
    #     {"insertText": {"location": {"index": 1}, "text": body}}]}).execute()
    # drive.permissions().create(fileId=doc_id,
    #     body={"type":"anyone","role":"reader"}).execute()
    # return f"https://docs.google.com/document/d/{doc_id}/edit"
    return f"https://docs.google.com/document/d/STUB_{abs(hash(title)) % 10000}/edit"


def write_link_to_excel(path: str, row_index: int, url: str) -> None:
    """Write url into the 'Blog link' cell (col D) and update date. Replace w/ openpyxl."""
    # wb = openpyxl.load_workbook(path); ws = wb.active
    # ws.cell(row=row_index, column=4, value=url)
    # ws.cell(row=row_index, column=3, value=datetime.date.today().strftime("%d/%m/%Y"))
    # wb.save(path)
    pass


# ---------- NODES ----------
def load_topics(state: AgentState) -> AgentState:
    state["rows"] = read_excel_rows(state["xlsx_path"])
    state["completed"], state["errors"] = [], []
    return state


def pick_next(state: AgentState) -> AgentState:
    state["current"] = state["rows"].pop(0) if state["rows"] else None
    state["draft"], state["doc_url"] = None, None
    return state


def write_blog(state: AgentState) -> AgentState:
    cur = state["current"]
    try:
        state["draft"] = llm_write_blog(cur["category"], cur["topic"])
    except Exception as e:
        state["errors"].append({"row": cur["row_index"], "error": f"write: {e}"})
        state["draft"] = None
    return state


def save_to_doc(state: AgentState) -> AgentState:
    cur = state["current"]
    if state["draft"]:
        try:
            state["doc_url"] = create_google_doc(cur["topic"], state["draft"])
        except Exception as e:
            state["errors"].append({"row": cur["row_index"], "error": f"doc: {e}"})
    return state


def update_sheet(state: AgentState) -> AgentState:
    cur = state["current"]
    if state["doc_url"]:
        write_link_to_excel(state["xlsx_path"], cur["row_index"], state["doc_url"])
        state["completed"].append({**cur, "doc_url": state["doc_url"]})
    return state


def finalize(state: AgentState) -> AgentState:
    print(f"Done. {len(state['completed'])} blogs created, "
          f"{len(state['errors'])} errors.")
    return state


# ---------- ROUTER ----------
def route_after_pick(state: AgentState) -> str:
    return "write_blog" if state["current"] is not None else "finalize"


# ---------- GRAPH CONSTRUCTION ----------
def build_graph():
    """
    g = StateGraph(AgentState)
    g.add_node("load_topics", load_topics)
    g.add_node("pick_next", pick_next)
    g.add_node("write_blog", write_blog)
    g.add_node("save_to_doc", save_to_doc)
    g.add_node("update_sheet", update_sheet)
    g.add_node("finalize", finalize)

    g.add_edge(START, "load_topics")
    g.add_edge("load_topics", "pick_next")
    g.add_conditional_edges("pick_next", route_after_pick,
                            {"write_blog": "write_blog", "finalize": "finalize"})
    g.add_edge("write_blog", "save_to_doc")
    g.add_edge("save_to_doc", "update_sheet")
    g.add_edge("update_sheet", "pick_next")   # <-- the loop back
    g.add_edge("finalize", END)
    return g.compile()
    """
    pass


# ---------- PLAIN-PYTHON SIMULATION (so this file runs without LangGraph) ----------
def run_simulation(xlsx_path: str = "topics.xlsx"):
    state: AgentState = {"xlsx_path": xlsx_path, "rows": [], "current": None,
                         "draft": None, "doc_url": None, "completed": [],
                         "errors": []}
    state = load_topics(state)
    while True:
        state = pick_next(state)
        if route_after_pick(state) == "finalize":
            break
        state = write_blog(state)
        state = save_to_doc(state)
        state = update_sheet(state)
        print(f"  row {state['current']['row_index']}: "
              f"{state['current']['topic']} -> {state['doc_url']}")
    finalize(state)
    return state


if __name__ == "__main__":
    run_simulation()
