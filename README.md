# 🎧 AI Customer Support Automation Desk
### Multi-Agent Support System with Microsoft AutoGen (v0.4), Hybrid FAISS RAG, Python & Streamlit

An enterprise-grade AI Customer Support Desk where a team of four specialized AI agents collaborates sequentially to triage incoming customer complaints, retrieve exact answers from company documents using **FAISS Vector Embeddings + Keyword Hybrid Search**, draft strictly grounded responses, enforce deterministic escalation rules, and log every ticket to CSV and Excel for audit and analytics.

---

## ✨ Key Features & Bonus Implementations

- **Four Single-Responsibility AutoGen v0.4 Agents**: Built with `autogen-agentchat` (`AssistantAgent` + `RoundRobinGroupChat`) running sequentially (`Query ➔ Knowledge ➔ Resolution ➔ Log`).
- **Multi-Format Hybrid FAISS RAG (Bonus)**: Automatically parses `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, and `.json` files in `knowledge_base/`, indexes chunks using `sentence-transformers` (`all-MiniLM-L6-v2`) and `faiss-cpu`, and applies a strict relevance threshold (`>= 0.42`) to prevent policy hallucinations.
- **Dynamic Knowledge Base Uploader**: Upload new PDF, Word, Markdown, or CSV policy documents directly from the Streamlit sidebar with automatic MD5 cache invalidation and FAISS re-indexing.
- **Deterministic Escalation Engine**: Automatically flags and escalates tickets involving high urgency, angry customers, missing knowledge base coverage, low confidence, or legal/fraud/billing disputes.
- **Idempotent Single-Ticket Audit Logger**: Persists every ticket to both `logs/tickets.csv` and `logs/tickets.xlsx` with built-in deduplication guards and instant `TextMentionTermination`.
- **Live Streamlit Timeline & Analytics Dashboard (Bonus)**: Displays real-time expandable agent steps, color-coded `Resolved` / `Escalated` badges, KPI metrics (resolution rate, escalation rate, high-urgency count), and one-click CSV/Excel downloads.

---

## 🏗️ System Architecture & Agent Workflow

```text
[Customer Complaint Submitted in Streamlit UI]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Customer_Query_Agent (agents/query_agent.py)         │
│    Extracts Category, Urgency, Sentiment, Language,     │
│    Legal/Fraud Risk & 1-line Summary.                   │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Knowledge_Base_Agent (agents/knowledge_agent.py)     │
│    Calls search_kb(query) -> Hybrid FAISS + Keyword     │
│    search across .md, .txt, .pdf, .docx, .csv files.    │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Resolution_Agent (agents/resolution_agent.py)        │
│    Drafts polite reply grounded ONLY in retrieved       │
│    passages. Never invents policies; outputs Confidence.│
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Log_Agent (agents/log_agent.py)                      │
│    Evaluates escalation rules, calls log_ticket(...)    │
│    to append 1 row to logs/tickets.csv & tickets.xlsx,  │
│    and emits TICKET_LOGGED_COMPLETE to terminate chat.  │
└─────────────────────────────────────────────────────────┘

## Testcases
1. Test Case #1 (Test #7) — 9 Days Late Order (Escalated Result)
Save your screenshot as screenshots/custom_test_7_escalated.png

2. Test Case #2 (Test #8) — Account Locked After 5 Tries (Resolved Result)
Save your screenshot as screenshots/custom_test_8_resolved.png