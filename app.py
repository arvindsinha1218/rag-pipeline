# pyrefly: ignore [missing-import]
import streamlit as st
from backend import handle_user_message

# ---------------- Page Config ----------------
st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")

# ---------------- Sidebar ----------------
def render_sidebar():
    st.sidebar.header("Navigation")

    if st.sidebar.button("🏠 Main Page", use_container_width=True):
        st.session_state.page = "main"

    if st.sidebar.button("💬 Chat", use_container_width=True):
        st.session_state.page = "chat"

    if st.sidebar.button("➕ New Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = None
        st.session_state.page = "chat"
        st.rerun()


# ---------------- Upload Section ----------------
def render_upload_section():
    st.header("📂 Add Resumes")

    files = st.file_uploader(
        "Select PDF resumes",
        type="pdf",
        accept_multiple_files=True
    )

    if files:
        st.success(f"{len(files)} file(s) selected")

        if st.button("Process & Index (Dummy)"):
            st.success("Resumes processed (mock)")


# ---------------- Search Section ----------------
def render_search_section():
    st.header("🔍 Search Candidates")

    with st.form("search_form"):
        job_title = st.text_input("Job Title")
        job_desc = st.text_area("Job Description")
        skills = st.text_area("Required Skills")
        top_k = st.slider("Results", 3, 10, 5)

        submitted = st.form_submit_button("Find Candidates")

    if submitted:
        st.success(f"Searching for {job_title} (mock results)")


# ---------------- CHAT PAGE (Agent Integrated) ----------------
def render_chat_page():
    st.header("💬 HireFlow Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_msg = st.chat_input("Ask about resumes...")

    if user_msg:
        # Show user msg
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg}
        )

        # 🔥 CALL YOUR AGENT HERE
        with st.spinner("Thinking..."):
            # Get existing session_id if present
            session_id = st.session_state.get("session_id")
        
            out = handle_user_message(
                user_msg,
                session_id=session_id
            )
        
            # Save session_id for future turns
            st.session_state.session_id = out["session_id"]
        
            bot_reply = out["response"]
        
        # Show bot reply
        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_reply}
        )

        st.rerun()


# ---------------- Main ----------------
def main():
    st.title("HireFlow")

    if "page" not in st.session_state:
        st.session_state.page = "main"

    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
    else:
        col1, col2 = st.columns(2)

        with col1:
            render_upload_section()

        with col2:
            render_search_section()


if __name__ == "__main__":
    main()
