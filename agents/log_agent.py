import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
from autogen_agentchat.agents import AssistantAgent

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CSV_PATH = LOG_DIR / "tickets.csv"
EXCEL_PATH = LOG_DIR / "tickets.xlsx"

COLUMNS = [
    "ticket_id",
    "timestamp",
    "customer_name / order_id",
    "complaint",
    "category",
    "urgency",
    "sentiment",
    "kb_sources",
    "response",
    "status",
    "escalation_reason",
]


def log_ticket(
    customer_info: str,
    complaint: str,
    category: str,
    urgency: str,
    sentiment: str,
    kb_sources: str,
    response: str,
    status: str,
    escalation_reason: str,
) -> str:
    """Appends a single processed support ticket row to logs/tickets.csv and logs/tickets.xlsx."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    clean_complaint = complaint.strip()
    next_num = 1

    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        try:
            with open(CSV_PATH, mode="r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                next_num = len(rows) + 1

                # DEDUPLICATION GUARD: If the exact same complaint was logged in the last 15 seconds, block duplicate write
                if rows:
                    last_row = rows[-1]
                    last_time = datetime.strptime(last_row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    seconds_diff = (datetime.now() - last_time).total_seconds()
                    if last_row["complaint"].strip() == clean_complaint and seconds_diff < 15:
                        return (
                            f"Ticket {last_row['ticket_id']} already recorded ({last_row['status']}). "
                            f"Final_Finish_Token: TICKET_LOGGED_COMPLETE"
                        )
        except Exception:
            next_num = 1

    ticket_id = f"TKT-{next_num:04d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data = {
        "ticket_id": ticket_id,
        "timestamp": timestamp,
        "customer_name / order_id": customer_info.strip() if customer_info.strip() else "Anonymous / N/A",
        "complaint": clean_complaint,
        "category": category.strip(),
        "urgency": urgency.strip(),
        "sentiment": sentiment.strip(),
        "kb_sources": kb_sources.strip(),
        "response": response.strip(),
        "status": status.strip(),
        "escalation_reason": escalation_reason.strip(),
    }

    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Keep Excel file synced
    try:
        df = pd.read_csv(CSV_PATH)
        df.to_excel(EXCEL_PATH, index=False)
    except Exception:
        pass

    # Returning TICKET_LOGGED_COMPLETE inside the tool output immediately triggers TextMentionTermination!
    return (
        f"✅ Ticket_ID: {ticket_id} | Status: {status} | "
        f"Escalation_Reason: {escalation_reason} | KB_Sources: {kb_sources} | "
        f"Final_Finish_Token: TICKET_LOGGED_COMPLETE"
    )


def create_log_agent(model_client) -> AssistantAgent:
    system_message = """You are the Log & Escalation Agent.
Review the conversation and call the `log_ticket` tool EXACTLY ONCE to record the ticket.

Mandatory Escalation Rule — Set `status="Escalated"` if ANY of the following is true:
1. Urgency is High
2. Sentiment is Angry
3. Knowledge Base Agent found no matching document (`kb_sources` is None or `NO_MATCHING_DOCUMENT_FOUND`)
4. Complaint involves legal threats (lawyer), fraud, or payment/billing disputes (e.g., charged multiple times)
5. Resolution Agent's Confidence is Low

If multiple escalation reasons apply, combine them into a single string (e.g., "High urgency + angry customer + legal threat") and call `log_ticket` ONLY ONCE.
Otherwise, set `status="Resolved"` and `escalation_reason="None"`."""

    return AssistantAgent(
        name="Log_Agent",
        description="Evaluates escalation rules and logs the ticket once to CSV/Excel via log_ticket.",
        model_client=model_client,
        tools=[log_ticket],
        reflect_on_tool_use=False,  # Ensures Log_Agent stops right after calling the tool once
        system_message=system_message,
    )