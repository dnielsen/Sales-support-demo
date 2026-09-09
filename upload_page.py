import io
from pathlib import Path

import streamlit as st
from docx import Document
from pypdf import PdfReader

from agent import ingest_file


def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".txt":
        return raw.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


def show_upload_page():
    st.set_page_config(page_title="Upload", page_icon="📤")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if "role" not in st.session_state:
        st.session_state.role = "guest"

    if not st.session_state.authenticated or st.session_state.role != "admin":
        st.info("Access denied.")
        return

    header_col, back_col, logout_col = st.columns([5, 1, 1])
    with header_col:
        st.title("Upload Documents")
    with back_col:
        if st.button("← Back", use_container_width=True, key="upload_back"):
            st.session_state.admin_view = "admin"
            st.rerun()
    with logout_col:
        if st.button("Logout", use_container_width=True, key="upload_logout_button"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.role = "guest"
            st.session_state.admin_view = "admin"
            st.rerun()

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size,
        }
        st.write(file_details)

        if st.button("Add to memory", key="upload_ingest_button"):
            progress = st.progress(0, text="Extracting text...")

            try:
                text = extract_text(uploaded_file)
            except Exception as e:
                progress.empty()
                st.error(f"Failed to extract text: {e}")
                st.stop()

            st.caption(f"Extracted {len(text)} characters")
            with st.expander("Preview extracted text"):
                st.text(text[:2000])

            if not text.strip():
                progress.empty()
                st.error(
                    "No text could be extracted from this file — it may be a "
                    "scanned/image-based PDF that pypdf can't read."
                )
                st.stop()

            try:
                ingest_file(
                    text,
                    on_progress=lambda pct, msg: progress.progress(pct, text=msg),
                )
                st.success(f"'{uploaded_file.name}' added to memory.")
            except Exception as e:
                progress.empty()
                st.error(f"Failed to ingest document: {e}")