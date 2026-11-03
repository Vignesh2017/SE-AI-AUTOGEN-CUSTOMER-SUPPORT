import asyncio
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from team import build_support_team
from agents.log_agent import CSV_PATH, EXCEL_PATH

st.set_page_config(
    page_title="AI Customer Support Desk | Multi-Agent System",
    page_icon="🎧",
    layout="wide",
)

# Custom styling for Resolved / Escalated badges and reply box
st.markdown(
    """
    <style>
    .badge-resolved {
        background-color: #16a34a;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-escalated {
        background-color: #dc2626;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .reply-card {
        background-color: #f8fafc;
        border-left: 6px solid #2563eb;
        padding: 18px 22px;
        border-radius: 8px;
        color: #0f172a;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎧 AI Customer Support Automation Desk")
st.caption("Powered by Microsoft AutoGen v0.4 Multi-Agent Pipeline (Query ➔ Knowledge RAG ➔ Resolution ➔ Escalation & Audit Logger)")

PRESET_TESTS = {
    "Select a sample test complaint...": ("", ""),
    "Test 1: Late Order (Shipping -> Resolved)": (
        "Arjun / ORD-1024",
        "Where is my order? It was supposed to arrive yesterday.",
    ),
    "Test 2: Shoe Return & Refund (Refund -> Resolved)": (
        "Meera / ORD-2089",
        "I want to return the shoes I bought last week. How do I get a refund?",
    ),
    "Test 3: Forgot Password (Account -> Resolved)": (
        "Karthik / N/A",
        "I forgot my password and can't log in.",
    ),
    "Test 4: Broken Blender Replacement (Product -> Resolved)": (
        "Divya / ORD-3391",
        "The blender I received is broken. Can I get a replacement?",
    ),
    "Test 5: Triple Card Charge + Lawyer Threat (Billing -> Escalated)": (
        "Vikram / ORD-4510",
        "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    ),
    "Test 6: Custom Branding 500 Gift Cards (Unmatched KB -> Escalated)": (
        "Priya Corp / N/A",
        "Do you sell gift cards for a company of 500 employees with custom branding?",
    ),
    "Bonus Test 7 (Custom): Order 9 Days Late & Furious (High Urgency -> Escalated)": (
        "Suresh / ORD-8821",
        "My order is 9 business days past the delivery date and tracking hasn't moved! I want my money back right now!",
    ),
    "Bonus Test 8 (Custom): Locked Account After 5 Tries (Account -> Resolved)": (
        "Ananya / N/A",
        "I typed my password wrong 5 times and now my account says it is locked. How do I unlock it quickly?",
    ),
}

with st.sidebar:
    st.header("⚡ Quick Test Cases")
    selected_preset = st.selectbox(
        "Load Assignment Test Case:",
        options=list(PRESET_TESTS.keys()),
    )
    preset_customer, preset_complaint = PRESET_TESTS[selected_preset]

    st.divider()
    st.subheader("📚 Active Knowledge Base")
    kb_folder = Path(__file__).resolve().parent / "knowledge_base"
    if kb_folder.exists():
        for doc in sorted(kb_folder.glob("*.md")):
            with st.expander(f"📄 {doc.name}"):
                st.markdown(doc.read_text(encoding="utf-8"))

col_form, col_workflow = st.columns([1, 1.3], gap="large")

with col_form:
    st.subheader("📝 Submit Customer Ticket")
    with st.form("complaint_form", clear_on_submit=False):
        customer_info = st.text_input(
            "Customer Name / Order ID (Optional)",
            value=preset_customer,
            placeholder="e.g., Priya / ORD-4512",
        )
        complaint_text = st.text_area(
            "Customer Complaint *",
            value=preset_complaint,
            height=140,
            placeholder="Describe your issue here...",
        )
        submitted = st.form_submit_button("🚀 Run Multi-Agent Support Desk", use_container_width=True, type="primary")


def parse_resolution_reply(text: str) -> str:
    match = re.search(r"Customer_Reply:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


async def run_pipeline(customer: str, complaint: str, step_container):
    team = build_support_team()
    task_prompt = (
        f"Customer Name / Order ID: {customer if customer else 'Anonymous / N/A'}\n"
        f"Raw Customer Complaint: {complaint}\n\n"
        f"Process this support ticket sequentially through Query -> Knowledge -> Resolution -> Log."
    )

    agent_outputs = {}
    agent_labels = {
        "Customer_Query_Agent": ("🔍 Step 1: Customer Query Agent (Triage & Sentiment)", "info"),
        "Knowledge_Base_Agent": ("📖 Step 2: Knowledge Base Agent (Document Grounding)", "warning"),
        "Resolution_Agent": ("✍️ Step 3: Resolution Agent (Grounded Draft)", "success"),
        "Log_Agent": ("🗂️ Step 4: Log & Escalation Agent (Audit & Routing)", "secondary"),
    }

    async for message in team.run_stream(task=task_prompt):
        source = getattr(message, "source", "")
        content = getattr(message, "content", "")

        if source in agent_labels and isinstance(content, str) and content.strip():
            # Keep most informative text message per agent
            agent_outputs[source] = content
            title, _ = agent_labels[source]
            with step_container.expander(title, expanded=True):
                st.markdown(content)

    return agent_outputs


with col_workflow:
    st.subheader("🤖 Live Multi-Agent Execution Timeline")
    timeline_box = st.container()

    if submitted:
        if not complaint_text.strip():
            st.error("Please enter a customer complaint before submitting.")
        else:
            with st.spinner("AutoGen agents are collaborating on this ticket..."):
                try:
                    outputs = asyncio.run(run_pipeline(customer_info, complaint_text, timeline_box))

                    res_text = outputs.get("Resolution_Agent", "")
                    log_text = outputs.get("Log_Agent", "")
                    final_reply = parse_resolution_reply(res_text)

                    is_escalated = "escalated" in log_text.lower() and "status: resolved" not in log_text.lower()
                    if CSV_PATH.exists():
                        df_latest = pd.read_csv(CSV_PATH)
                        if not df_latest.empty:
                            last_row = df_latest.iloc[-1]
                            is_escalated = str(last_row["status"]).strip().lower() == "escalated"
                            final_reply = str(last_row["response"])
                            esc_reason = str(last_row["escalation_reason"])
                            kb_src = str(last_row["kb_sources"])
                            tkt_id = str(last_row["ticket_id"])
                        else:
                            esc_reason, kb_src, tkt_id = "N/A", "N/A", "TKT-0001"
                    else:
                        esc_reason, kb_src, tkt_id = "N/A", "N/A", "TKT-0001"

                    st.divider()
                    st.subheader("📬 Final Customer Response")

                    badge_html = (
                        f'<span class="badge-escalated">🚨 ESCALATED TO HUMAN SUPPORT ({tkt_id})</span>'
                        if is_escalated
                        else f'<span class="badge-resolved">✅ RESOLVED AUTOMATICALLY ({tkt_id})</span>'
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)

                    if is_escalated:
                        st.warning(f"**Escalation Trigger:** {esc_reason} | **KB Grounding:** `{kb_src}`")
                    else:
                        st.caption(f"**KB Sources Used:** `{kb_src}`")

                    st.markdown(f'<div class="reply-card">{final_reply}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Pipeline Execution Error: {e}")

st.divider()

# Section: Ticket Log & Bonus Analytics Dashboard
st.subheader("📊 Ticket Log & Support Analytics")

if CSV_PATH.exists():
    df_logs = pd.read_csv(CSV_PATH)
    if not df_logs.empty:
        m1, m2, m3, m4 = st.columns(4)
        total_tickets = len(df_logs)
        resolved_count = len(df_logs[df_logs["status"].astype(str).str.lower() == "resolved"])
        escalated_count = len(df_logs[df_logs["status"].astype(str).str.lower() == "escalated"])
        high_urgency_count = len(df_logs[df_logs["urgency"].astype(str).str.lower() == "high"])

        m1.metric("Total Tickets Logged", total_tickets)
        m2.metric("Resolved Automatically", resolved_count, f"{(resolved_count/total_tickets)*100:.0f}% rate")
        m3.metric("Escalated to Human", escalated_count, f"{(escalated_count/total_tickets)*100:.0f}% rate", delta_color="inverse")
        m4.metric("High Urgency Cases", high_urgency_count)

        st.dataframe(df_logs, use_container_width=True, hide_index=True)

        dl_col1, dl_col2 = st.columns([1, 5])
        with dl_col1:
            csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV Log",
                data=csv_bytes,
                file_name="tickets.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            if EXCEL_PATH.exists():
                with open(EXCEL_PATH, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Excel (.xlsx) Log",
                        data=f.read(),
                        file_name="tickets.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
else:
    st.info("No tickets logged yet. Submit a complaint above to generate `logs/tickets.csv`.")