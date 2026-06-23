import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool

# ──────────────────────────────────────────────────────────────────────────────
# WHY THIS FILE EXISTS
#
# The Researcher agent needs a tool to look up information.
# This file does three things:
#   1. Stores the TechFlow FAQ as a hardcoded string constant
#   2. Builds a FAISS vector index from that FAQ (same RAG pipeline as Project 5)
#   3. Wraps the FAISS search in a LangChain Tool that CrewAI agents can call
#
# The key difference from Project 5: there is no user-uploaded PDF here.
# The knowledge base is always the same hardcoded FAQ, so we build it once
# at app startup and never rebuild it.
# ──────────────────────────────────────────────────────────────────────────────


# ── TechFlow CRM Knowledge Base ────────────────────────────────────────────────
# This is the "database" the Researcher agent searches.
# Written as a single multi-section string so it can be chunked like a document.
# Covers all 4 query categories: billing, technical, complaint (policies), general.
TECHFLOW_FAQ = """
TechFlow CRM — Complete Knowledge Base and Support FAQ

== ABOUT TECHFLOW ==
TechFlow is a cloud-based CRM (Customer Relationship Management) platform for small to
medium-sized sales teams. It helps businesses manage contacts, track deals through sales
pipelines, automate email outreach, and generate sales reports. Founded in 2019, TechFlow
serves over 15,000 businesses globally.

== PLANS AND PRICING ==

STARTER PLAN — $29 per user per month (or $290 per user per year — save 17%)
- Up to 2,500 contacts
- 1 sales pipeline
- Basic email integration (Gmail and Outlook)
- Standard reporting (5 pre-built reports)
- Email support only with 48-hour response time
- 5 GB storage per user
- Maximum 5 users

PROFESSIONAL PLAN — $79 per user per month (or $790 per user per year — save 17%)
- Up to 25,000 contacts
- Unlimited sales pipelines
- Advanced email integration with tracking and automated sequences
- Custom reporting and dashboards
- Priority live chat and email support with 4-hour response time
- 50 GB storage per user
- Unlimited users
- API access (10,000 calls per day)
- Slack and Microsoft Teams integration
- Workflow automation (up to 50 active workflows)
- Mobile app for iOS and Android

ENTERPRISE PLAN — $199 per user per month (or $1,990 per user per year — save 17%)
- Unlimited contacts
- All Professional features included
- Unlimited workflow automation
- Dedicated account manager
- Phone support available 24/7
- 500 GB storage per user
- Unlimited API calls
- Custom integrations
- SSO (Single Sign-On) with SAML 2.0
- IP whitelisting and full audit logs
- 99.9% uptime SLA guarantee
- Custom onboarding and training sessions
- Data residency options: US, EU, or APAC regions
- No free trial available for Enterprise — contact sales@techflow.io for a demo

FREE TRIAL:
- A 14-day free trial is available for the Starter and Professional plans
- No credit card required to start the trial
- Trial includes full access to all plan features
- At the end of the trial, enter payment details to continue or the account is paused
- Trial data is preserved for 30 days after trial expiry

== BILLING AND PAYMENTS ==

BILLING CYCLES:
- Monthly billing: charged on the same date each month (your billing date is set on
  the day you first subscribed)
- Annual billing: charged once per year upfront. Annual plans save 17% vs monthly.
- Invoices are emailed automatically within 1 hour of each payment to the billing email

PAYMENT METHODS ACCEPTED:
- Visa, Mastercard, American Express, Discover
- PayPal
- Bank transfer (Enterprise plans only, minimum 10-seat commitment required)
- Prepaid cards and gift cards are not accepted

CHANGING PLANS:
- Upgrades: take effect immediately. You are charged a prorated amount for the
  remainder of the current billing period.
- Downgrades: take effect at the start of your next billing period. You keep the
  higher plan features until then.
- To change plan: Settings > Billing > Change Plan

REFUND POLICY:
- Monthly plans: refund available within 7 days of a charge if requested for the
  first time. After 7 days, no refund for the current month.
- Annual plans: full refund within 30 days of purchase. After 30 days, prorated
  refund for unused months minus a 10% processing fee.
- To request a refund: email billing@techflow.io with your account email and charge date
- Refunds are processed within 5-7 business days to your original payment method

DUPLICATE CHARGES AND BILLING ERRORS:
- A duplicate charge usually occurs when a payment fails and the system retries,
  but the original charge also processed successfully.
- TechFlow will fully refund any confirmed duplicate charges within 3-5 business days.
- To report a duplicate charge or billing error: email billing@techflow.io or go to
  Settings > Billing > Report an Issue
- Please include: your account email, the charge amounts, and the dates on your statement
- Our billing team responds to all billing inquiries within 4 business hours

CANCELLATION POLICY:
- Cancel anytime from Settings > Billing > Cancel Subscription
- Monthly plans: access continues until end of current billing period, no refund for
  remaining days
- Annual plans: prorated refund for unused months (minus 10% fee) if cancelled after
  the 30-day full refund window
- After cancellation, data is retained for 60 days then permanently deleted
- To reactivate within 60 days: log in and go to Settings > Billing > Reactivate

FAILED PAYMENTS:
- TechFlow retries failed payments 3 times over 7 days
- After 3 failures, the account is suspended (all data is preserved for 30 days)
- To reactivate a suspended account: update payment method in Settings > Billing >
  Update Payment Method

== EMAIL INTEGRATION ==

SUPPORTED EMAIL PROVIDERS:
- Gmail (Google Workspace and personal Gmail accounts)
- Microsoft Outlook (Office 365 and Outlook.com)
- Other providers via IMAP/SMTP (manual configuration required)

GMAIL INTEGRATION SETUP:
1. Go to Settings > Integrations > Email > Connect Gmail
2. Click Sign in with Google and authorize TechFlow
3. Select which Gmail account to connect
4. Initial sync takes 15-30 minutes for inboxes with large history
5. After setup, emails sent and received with contacts sync automatically

GMAIL SYNC ISSUES AND TROUBLESHOOTING:
- If sync stops working, disconnect and reconnect: Settings > Integrations > Email >
  Disconnect > Reconnect Gmail
- Common causes of Gmail sync failure: expired Google authorization token, changed
  Google password, revoked app permissions in Google account settings
- Check integration status: Settings > Integrations > Email shows Connected (green)
  or Error (red) with error details
- If reconnecting does not fix the issue: check TechFlow is still authorized at
  myaccount.google.com/permissions — if TechFlow is not listed, reconnect
- Sync delay: newly sent or received emails appear in TechFlow within 5-10 minutes
- Gmail sync does not work in real-time — there is always a short delay

OUTLOOK INTEGRATION:
- Same setup process as Gmail but uses Microsoft OAuth authentication
- If Outlook sync fails: check that your Microsoft 365 admin has not blocked
  third-party app connections (common in enterprise accounts)
- Enterprise Microsoft 365 accounts may require admin approval before connecting

== CONTACT AND PIPELINE MANAGEMENT ==

CONTACT LIMITS BY PLAN:
- Starter: 2,500 contacts maximum
- Professional: 25,000 contacts maximum
- Enterprise: unlimited contacts
- Warning appears at 90% capacity; upgrade plan or archive unused contacts to free space

IMPORTING CONTACTS:
- CSV import: Contacts > Import > Upload CSV
- Required columns: at least one of first_name or last_name, plus email address
- Optional columns: phone, company, job_title, address, custom fields
- Maximum file size per import: 50 MB
- Duplicate detection is automatic based on email address

PIPELINE STAGES (default):
Lead > Qualified > Proposal > Negotiation > Closed Won > Closed Lost
- Custom stages: add, rename, or remove in Settings > Pipelines
- Moving deals: drag and drop in Kanban view

DATA LOSS AND RECOVERY:
- Deleted contacts, deals, and data go to the Recycle Bin
- Recycle Bin: Settings > Data > Recycle Bin
- Items in Recycle Bin can be restored within 30 days before permanent deletion
- For accidental bulk deletion or major data loss: contact support immediately at
  support@techflow.io — data recovery from daily backups is possible within 30 days
- TechFlow performs daily automatic backups; backups are retained for 30 days

== INTEGRATIONS AND API ==

AVAILABLE NATIVE INTEGRATIONS:
- Gmail and Outlook (email sync)
- Slack and Microsoft Teams (deal and task notifications)
- Google Calendar and Outlook Calendar (meeting sync)
- LinkedIn Sales Navigator (Professional and Enterprise only)
- Zapier (connect to 5,000+ other apps, all plans)
- Stripe for payment data sync (Enterprise only)
- QuickBooks and Xero for invoice sync (Enterprise only)

API ACCESS:
- Available on Professional plan (10,000 calls/day) and Enterprise (unlimited)
- Not available on Starter plan
- API key: Settings > API > Generate Key
- Documentation: developers.techflow.io
- Rate limit: 100 requests per minute per API key

== ACCOUNT MANAGEMENT ==

USER ROLES:
- Admin: full access including billing, settings, and user management
- Manager: view and edit all team data, manage pipelines, create reports — no billing access
- User: manage own contacts, deals, and tasks — limited reporting
- Read-only: view all data, no create or edit permissions

ADDING AND REMOVING USERS:
- Add: Settings > Team > Invite User (enter email, select role, send invitation)
- Invitations expire after 7 days
- Remove: Settings > Team > select user > Deactivate (their data is reassigned to admin)
- Billing: new users billed prorated; removed users receive a credit for remaining days

DATA BACKUP AND EXPORT:
- Export all data: Settings > Data > Export All Data (CSV format)
- Large exports may take up to 30 minutes
- TechFlow performs automatic daily backups retained for 30 days
- Contact support within 30 days for backup-based data recovery

== COMMON TROUBLESHOOTING ==

LOGIN PROBLEMS:
- Forgot password: click Forgot Password on the login page and check email
- Account locked after 5 failed attempts: locked for 30 minutes, then automatically unlocked
- Browser issues: try clearing cache and cookies, or use an incognito window

PERFORMANCE ISSUES:
- Check status.techflow.io for any ongoing incidents
- Try a hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Large exports or report generation running in background can slow the interface temporarily

MOBILE APP:
- iOS: search TechFlow CRM in the App Store
- Android: search TechFlow CRM in Google Play
- Push notifications: check notification permissions in phone settings
- If app is not syncing: force close and reopen; ensure you have the latest version

== SUPPORT CONTACTS ==
- General support: support@techflow.io
- Billing issues: billing@techflow.io
- Security issues: security@techflow.io
- Privacy and GDPR: privacy@techflow.io
- Sales and Enterprise inquiries: sales@techflow.io
- Status page: status.techflow.io
- Help centre: help.techflow.io
- Developer documentation: developers.techflow.io
- Response times: Starter 48 hours (email only), Professional 4 hours (chat + email),
  Enterprise 24/7 phone support with dedicated account manager

== SECURITY AND COMPLIANCE ==
- All data encrypted in transit (TLS 1.2+) and at rest (AES-256)
- SOC 2 Type II certified
- GDPR compliant; Data Processing Agreement available at legal@techflow.io
- EU data residency available on Enterprise plan
- Two-factor authentication available on all plans: Settings > Security > Enable 2FA
- Enterprise SSO with SAML 2.0
"""


@st.cache_resource
def get_vector_store() -> FAISS:
    """
    Build and return a FAISS vector store from the TechFlow FAQ.

    @st.cache_resource means this function runs ONCE per app session.
    The embedding model (~80 MB) downloads on first run, then stays cached.

    Why FAISS here instead of a database?
    The FAQ is static — it never changes between queries.
    FAISS in-memory is fast, free, and requires no external service.
    The index fits comfortably in Streamlit Cloud's 1 GB RAM.

    Returns:
        A FAISS vector store ready for similarity_search()
    """
    # Step 1: Load the embedding model (same model as Project 5)
    # all-MiniLM-L6-v2 converts text to 384-dimensional vectors.
    # Similar meaning = similar vectors = found by similarity search.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 2: Wrap the FAQ text in a LangChain Document object.
    # The metadata here just labels the source — useful for debugging.
    faq_document = Document(
        page_content=TECHFLOW_FAQ,
        metadata={"source": "TechFlow CRM Knowledge Base"}
    )

    # Step 3: Split into overlapping chunks.
    # Same settings as Project 5 — 1000 chars per chunk, 200 overlap.
    # The splitter copies the metadata to every child chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents([faq_document])

    # Step 4: Embed all chunks and build the FAISS index.
    # FAISS.from_documents() does both in one call.
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def build_rag_tool(vector_store: FAISS) -> Tool:
    """
    Wrap the FAISS vector store in a LangChain Tool that CrewAI agents can call.

    Why a Tool wrapper?
    CrewAI agents work like ReAct agents — they decide WHEN to call a tool
    and WHAT query to pass. The tool's name and description tell the agent
    what the tool does and when to use it. The agent reads these to decide.

    The Researcher agent has this tool in its tools list.
    The other 3 agents do not — they don't need to search the knowledge base.

    Args:
        vector_store: The FAISS index built by get_vector_store()

    Returns:
        A LangChain Tool object that performs semantic search on the FAQ
    """

    def search_faq(query: str) -> str:
        """
        Inner function that actually runs the FAISS search.
        Called by the Tool when the Researcher agent uses it.
        Returns the top 3 most relevant chunks as a single string.
        """
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the TechFlow knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[Result {i}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    # Tool definition — the name and description are what the LLM reads
    # to decide when and how to call this tool.
    # Make the description specific: "when to use" and "what it returns".
    return Tool(
        name="TechFlow Knowledge Base Search",
        description=(
            "Search TechFlow CRM's official knowledge base and FAQ documentation. "
            "Use this tool to find information about: pricing plans, billing policies, "
            "refund procedures, email integration setup and troubleshooting, contact limits, "
            "pipeline management, API access, integrations, account settings, and support policies. "
            "Input should be a specific search query. Returns relevant documentation sections."
        ),
        func=search_faq,
    )
