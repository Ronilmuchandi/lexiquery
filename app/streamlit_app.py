# =============================================================
# FILE: app/streamlit_app.py
# PURPOSE: The frontend of LexiQuery — a beautiful web app
#          where users can upload contracts and get instant
#          AI-powered legal analysis
#
# HOW TO RUN:
#   streamlit run app/streamlit_app.py
#
# WHAT IT DOES:
#   - Upload PDF contracts
#   - Ask questions in plain English
#   - Get AI answers with clause citations
#   - See risk score (1-10) with visual indicator
#   - Compare multiple contracts
# =============================================================
import sys
import os

# Add project root to Python path
# Fixes 'No module named src' error on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import os
import tempfile

# Our pipeline functions
from src.pipeline.indexer import process_and_index_contract, get_indexed_contracts
from src.pipeline.analyzer import answer_legal_question, flag_risky_clauses
from src.pipeline.comparator import score_contract_risk, compare_contracts

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# =============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command
# =============================================================
st.set_page_config(
    page_title="LexiQuery — Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================
# CUSTOM CSS
# Makes the app look professional and clean
# =============================================================
st.markdown("""
<style>
    /* Background image - law themed */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=1920');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Dark overlay for readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(10, 15, 30, 0.85);
        z-index: 0;
    }
    
    /* Main header */
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #c9a84c;
    text-align: center;
    padding: 1rem 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
}
    
    /* Subtitle */
    .sub-header {
        font-size: 1.1rem;
        color: #e0c97f;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Risk score colors */
    .risk-low { color: #28a745; font-size: 2rem; font-weight: 700; }
    .risk-medium { color: #ffc107; font-size: 2rem; font-weight: 700; }
    .risk-high { color: #dc3545; font-size: 2rem; font-weight: 700; }
    
    /* Clause card */
    .clause-card {
        background: rgba(255,255,255,0.08);
        border-left: 4px solid #c9a84c;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #fff;
    }
    
    /* Answer box */
    .answer-box {
        background: rgba(201, 168, 76, 0.15);
        border: 1px solid #c9a84c;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #fff;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #c9a84c;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #c9a84c44;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def get_risk_color(score: int) -> str:
    """Return color class based on risk score."""
    if score <= 3:
        return "risk-low"
    elif score <= 6:
        return "risk-medium"
    else:
        return "risk-high"


def get_risk_emoji(score: int) -> str:
    """Return emoji based on risk score."""
    if score <= 3:
        return "🟢"
    elif score <= 6:
        return "🟡"
    else:
        return "🔴"


def display_risk_score(result: dict):
    """Display risk score in a visual way."""
    score = result["risk_score"]
    level = result["risk_level"]
    emoji = get_risk_emoji(score)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem;
        background: #f8f9fa; border-radius: 12px;
        border: 2px solid #dee2e6;'>
            <div style='font-size: 1rem; color: #666;
            margin-bottom: 0.5rem;'>Overall Risk Score</div>
            <div style='font-size: 3.5rem;'>{emoji}</div>
            <div class='{get_risk_color(score)}'>{score}/10</div>
            <div style='font-size: 1.1rem; color: #444;
            margin-top: 0.5rem;'>{level} Risk</div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar for visual representation
    st.progress(score / 10)

    # Summary
    if result.get("summary"):
        st.info(f"📋 {result['summary']}")

    # Red flags
    if result.get("red_flags"):
        st.subheader("🚩 Red Flags")
        for flag in result["red_flags"]:
            if flag.strip():
                st.warning(f"⚠️ {flag}")

    # Recommendations
    if result.get("recommendations"):
        st.subheader("💡 Recommendations")
        for rec in result["recommendations"]:
            if rec.strip():
                st.success(f"✅ {rec}")


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=60)
    st.title("LexiQuery")
    st.caption("AI-Powered Legal Contract Analyzer")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📄 Analyze Contract", "⚖️ Risk Score", "🔍 Compare Contracts"],
        label_visibility="collapsed"
    )

    st.divider()

    # Show indexed contracts
    st.subheader("📁 Indexed Contracts")
    contracts = get_indexed_contracts()
    if contracts:
        for c in contracts:
            st.success(f"✅ {c}")
    else:
        st.info("No contracts indexed yet")

    st.divider()
    st.caption("Built with LangChain, ChromaDB, Groq & Streamlit")


# =============================================================
# HOME PAGE
# =============================================================
if page == "🏠 Home":
    st.markdown(
        '<div class="main-header">⚖️ LexiQuery</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Intelligent Legal Contract Analysis powered by AI</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div style='background:rgba(255,255,255,0.15);
        padding:1rem;border-radius:8px;border:1px solid #c9a84c;'>
        <span style='color:#c9a84c;font-size:1.2rem;font-weight:700;'>📄 Upload</span>
        <p style='color:white;margin-top:0.5rem;'>Upload any legal contract PDF and let LexiQuery index it instantly</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div style='background:rgba(255,255,255,0.15);
        padding:1rem;border-radius:8px;border:1px solid #c9a84c;'>
        <span style='color:#c9a84c;font-size:1.2rem;font-weight:700;'>💬 Ask</span>
        <p style='color:white;margin-top:0.5rem;'>Ask questions in plain English — no legal knowledge needed</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div style='background:rgba(255,255,255,0.15);
        padding:1rem;border-radius:8px;border:1px solid #c9a84c;'>
        <span style='color:#c9a84c;font-size:1.2rem;font-weight:700;'>🎯 Score</span>
        <p style='color:white;margin-top:0.5rem;'>Get an instant risk score from 1-10 with detailed red flags</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div style='background:rgba(255,255,255,0.15);
        padding:1rem;border-radius:8px;border:1px solid #c9a84c;'>
        <span style='color:#c9a84c;font-size:1.2rem;font-weight:700;'>🔍 Compare</span>
        <p style='color:white;margin-top:0.5rem;'>Compare multiple contracts side by side to find the best deal</p>
        </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🚀 How to Get Started")
    st.markdown("""
    1. Click **Analyze Contract** in the sidebar
    2. Upload your PDF contract
    3. Ask any question about the contract
    4. Check the **Risk Score** tab for a full risk analysis
    """)

    st.markdown(
        '<div class="footer">LexiQuery — Built by Ronil Muchandi | MS Data Science, University of Missouri</div>',
        unsafe_allow_html=True
    )


# =============================================================
# ANALYZE CONTRACT PAGE
# =============================================================
elif page == "📄 Analyze Contract":
    st.title("📄 Analyze Contract")
    st.caption("Upload a contract and ask questions in plain English")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your contract (PDF)",
        type=["pdf"],
        help="Upload any legal contract, NDA, employment agreement, etc."
    )

    if uploaded_file:
        # Get contract name from file name
        contract_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_")

        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Index button
        if st.button("🔍 Index this Contract", type="primary"):
            with st.spinner("Reading and indexing contract..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process and index
                result = process_and_index_contract(tmp_path, contract_name)

                # Clean up temp file
                os.unlink(tmp_path)

                if result["success"]:
                    st.success(f"""
                    ✅ Contract indexed successfully!
                    - Total clauses found: {result['total_clauses']}
                    - Total characters: {result['total_characters']}
                    """)
                    st.session_state["current_contract"] = contract_name
                else:
                    st.error(f"❌ Error: {result['error']}")

    st.divider()

    # Q&A Section
    st.subheader("💬 Ask a Question")

    # Contract selector
    contracts = get_indexed_contracts()
    if contracts:
        selected_contract = st.selectbox(
            "Select contract to query",
            ["All contracts"] + contracts
        )

        # Example questions
        st.caption("Example questions:")
        example_cols = st.columns(2)
        with example_cols[0]:
            if st.button("What are my obligations?"):
                st.session_state["question"] = "What are my obligations under this agreement?"
            if st.button("Are there termination clauses?"):
                st.session_state["question"] = "What are the termination conditions?"
        with example_cols[1]:
            if st.button("What are the risky clauses?"):
                st.session_state["question"] = "Flag any clauses that are risky or unusual"
            if st.button("What is confidential?"):
                st.session_state["question"] = "What information is considered confidential?"

        # Question input
        question = st.text_area(
            "Your question",
            value=st.session_state.get("question", ""),
            placeholder="e.g. What happens if I break this agreement?",
            height=100
        )

        if st.button("🔍 Get Answer", type="primary"):
            if question:
                with st.spinner("Analyzing contract..."):
                    contract_filter = (
                        None if selected_contract == "All contracts"
                        else selected_contract
                    )
                    result = answer_legal_question(question, contract_filter)

                # Display answer
                st.subheader("📋 Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

                # Show sources
                st.subheader("📎 Sources Used")
                for source in result["clauses_used"]:
                    st.caption(f"• {source}")

                # Show relevant clauses
                with st.expander("🔍 View Relevant Clauses"):
                    for clause in result["relevant_clauses"]:
                        st.markdown(f"""
                        <div class="clause-card">
                            <strong>Clause {clause['clause_number']} 
                            from {clause['contract']}</strong>
                            <br>Relevance: {clause['relevance_score']}
                            <br><br>{clause['text']}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a question!")
    else:
        st.info("👆 Please upload and index a contract first!")


# =============================================================
# RISK SCORE PAGE
# =============================================================
elif page == "⚖️ Risk Score":
    st.title("⚖️ Contract Risk Score")
    st.caption("Get an AI-powered risk assessment for any indexed contract")

    contracts = get_indexed_contracts()

    if contracts:
        selected = st.selectbox("Select contract to score", contracts)

        if st.button("🎯 Analyze Risk", type="primary"):
            with st.spinner("Analyzing contract risk..."):
                result = score_contract_risk(selected)

            if result["success"]:
                display_risk_score(result)
            else:
                st.error(f"❌ {result['error']}")
    else:
        st.info("👆 Please upload and index a contract first!")


# =============================================================
# COMPARE CONTRACTS PAGE
# =============================================================
elif page == "🔍 Compare Contracts":
    st.title("🔍 Compare Contracts")
    st.caption("Compare multiple contracts side by side")

    contracts = get_indexed_contracts()

    if len(contracts) >= 2:
        selected = st.multiselect(
            "Select contracts to compare (minimum 2)",
            contracts
        )

        if len(selected) >= 2:
            if st.button("🔍 Compare", type="primary"):
                with st.spinner("Comparing contracts..."):
                    result = compare_contracts(selected)

                if result["success"]:
                    st.subheader("📊 Individual Risk Scores")
                    cols = st.columns(len(selected))
                    for i, name in enumerate(selected):
                        with cols[i]:
                            score = result["individual_scores"][name]["risk_score"]
                            emoji = get_risk_emoji(score)
                            st.metric(
                                label=name,
                                value=f"{emoji} {score}/10"
                            )

                    st.subheader("📋 Comparison Analysis")
                    st.write(result["comparison_analysis"])
                else:
                    st.error(f"❌ {result['error']}")
        else:
            st.info("Please select at least 2 contracts")
    elif len(contracts) == 1:
        st.info("👆 Please index at least one more contract to compare!")
    else:
        st.info("👆 Please upload and index contracts first!")