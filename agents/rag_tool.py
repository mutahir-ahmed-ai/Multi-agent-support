import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool

TECHFLOW_FAQ = """
TechFlow CRM — Complete Knowledge Base and Support FAQ

== ABOUT TECHFLOW ==
TechFlow is a cloud-based CRM platform for small to medium-sized sales teams.
It helps businesses manage contacts, track deals through sales pipelines,
automate email outreach, and generate sales reports. Founded in 2019,
TechFlow serves over 15,000 businesses globally.

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
- Unlimited contacts and pipelines
- All Professional features included
- Unlimited workflow automation
- Dedicated account manager
- Phone support available 24/7
- 500 GB storage per user
- Unlimited API calls
- SSO (Single Sign-On) with SAML 2.0
- IP whitelisting and full audit logs
- 99.9% uptime SLA guarantee
- Custom onboarding and training sessions
- Data residency options: US, EU, or APAC regions

FREE TRIAL:
- 14-day free trial available for Starter and Professional plans
- No credit card required to start the trial
- Trial data preserved for 30 days after expiry

== BILLING AND PAYMENTS ==

BILLING CYCLES:
- Monthly billing: charged on the same date each month
- Annual billing: charged once per year upfront, saves 17% vs monthly
- Invoices emailed automatically within 1 hour of payment

PAYMENT METHODS ACCEPTED:
- Visa, Mastercard, American Express, Discover
- PayPal
- Bank transfer (Enterprise plans only, minimum 10-seat commitment)
- Prepaid cards and gift cards are not accepted

CHANGING PLANS:
- Upgrades: take effect immediately, charged prorated amount for remainder of billing period
- Downgrades: take effect at start of next billing period
- To change plan: Settings > Billing > Change Plan

REFUND POLICY:
- Monthly plans: refund available within 7 days of a charge if requested for the first time
- Annual plans: full refund within 30 days of purchase. After 30 days, prorated refund
  for unused months minus a 10% processing fee
- To request a refund: email billing@techflow.io with your account email and charge date
- Refunds processed within 5-7 business days to your original payment method

DUPLICATE CHARGES AND BILLING ERRORS:
- A duplicate charge usually occurs when a payment fails and the system retries
  but the original charge also processed successfully
- TechFlow will fully refund any confirmed duplicate charges within 3-5 business days
- To report: email billing@techflow.io or go to Settings > Billing > Report an Issue
- Include: your account email, the charge amounts, and the dates on your statement
- Billing team responds to all billing inquiries within 4 business hours

CANCELLATION POLICY:
- Cancel anytime from Settings > Billing > Cancel Subscription
- Monthly plans: access continues until end of current billing period
- Annual plans: prorated refund for unused months minus 10% fee after 30-day window
- After cancellation, data retained for 60 days then permanently deleted

FAILED PAYMENTS:
- TechFlow retries failed payments 3 times over 7 days
- After 3 failures, account is suspended (data preserved for 30 days)
- To reactivate: update payment method in Settings > Billing > Update Payment Method

== EMAIL INTEGRATION ==

GMAIL INTEGRATION SETUP:
1. Go to Settings > Integrations > Email > Connect Gmail
2. Click Sign in with Google and authorize TechFlow
3. Select which Gmail account to connect
4. Initial sync takes 15-30 minutes for large inboxes
5. After setup, emails sent and received with contacts sync automatically

GMAIL SYNC TROUBLESHOOTING:
- If sync stops: disconnect and reconnect at Settings > Integrations > Email
- Common causes: expired Google auth token, changed password, revoked permissions
- Check status: Settings > Integrations > Email shows Connected (green) or Error (red)
- If reconnecting fails: verify TechFlow is authorized at myaccount.google.com/permissions
- Sync delay: new emails appear in TechFlow within 5-10 minutes (not real-time)

OUTLOOK INTEGRATION:
- Same setup as Gmail but uses Microsoft OAuth
- Enterprise accounts may require admin approval before connecting

== CONTACT AND PIPELINE MANAGEMENT ==

CONTACT LIMITS:
- Starter: 2,500 contacts maximum
- Professional: 25,000 contacts maximum
- Enterprise: unlimited contacts
- Warning at 90% capacity; upgrade or archive unused contacts

DATA LOSS AND RECOVERY:
- Deleted data goes to Recycle Bin: Settings > Data > Recycle Bin
- Items restorable within 30 days before permanent deletion
- For major data loss: contact support@techflow.io immediately
- Daily automatic backups retained for 30 days

== SUPPORT CONTACTS ==
- General support: support@techflow.io
- Billing issues: billing@techflow.io
- Status page: status.techflow.io
- Help centre: help.techflow.io
- Response times: Starter 48 hours email, Professional 4 hours chat+email,
  Enterprise 24/7 phone with dedicated account manager
"""


@st.cache_resource
def get_vector_store() -> FAISS:
    """Build FAISS index from the TechFlow FAQ. Cached — runs once per session."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    faq_document = Document(
        page_content=TECHFLOW_FAQ,
        metadata={"source": "TechFlow CRM Knowledge Base"}
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents([faq_document])
    return FAISS.from_documents(chunks, embeddings)


def build_rag_tool(vector_store: FAISS) -> Tool:
    """Wrap FAISS in a LangChain Tool that the CrewAI Researcher agent can call."""

    def search_faq(query: str) -> str:
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the TechFlow knowledge base."
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[Result {i}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    return Tool(
        name="TechFlow Knowledge Base Search",
        description=(
            "Search TechFlow CRM's official knowledge base and FAQ. "
            "Use this to find: pricing plans, billing policies, refund procedures, "
            "email integration troubleshooting, contact limits, API access, "
            "integrations, account settings, and support policies. "
            "Input should be a specific search query."
        ),
        func=search_faq,
    )
