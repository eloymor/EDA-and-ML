import os
import streamlit as st
from resume_analysis import extract_info, scan_resume


APP_TITLE = "Resume Analyzer"
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
MODELS = ["qwen3:4b_ollama", "gemma3:4b_ollama", "mistral:7b_ollama", "qwen3:4b_transformers"]


def init_session_state():
    # No chat state needed
    if "resume_md" not in st.session_state:
        st.session_state.resume_md: str | None = None
    if "resume_path" not in st.session_state:
        st.session_state.resume_path: str | None = None
    if "model" not in st.session_state:
        st.session_state.model = "qwen3:4b_ollama"
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE


def sidebar_controls():
    with st.sidebar:
        st.header("Settings")
        st.caption("Choose model and load a resume PDF.")
        model_type = st.selectbox("Model", options=MODELS, index=MODELS.index(st.session_state.model))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(st.session_state.temperature), step=0.1)
        uploaded = st.file_uploader("Upload resume PDF", type=["pdf"], accept_multiple_files=False)

        if st.button("Load Resume", type="primary"):
            if uploaded is None:
                st.warning("Please upload a PDF file first.")
            else:
                # Save to a temporary file path for scan_resume
                try:
                    tmp_dir = st.query_params.get("tmp_dir", [None])[0]  # type: ignore[attr-defined]
                except Exception:
                    tmp_dir = None
                if not tmp_dir:
                    tmp_dir = os.path.join(os.getcwd(), ".streamlit_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, uploaded.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                # Try converting now to validate and allow preview/use in legacy flows
                try:
                    resume_md = extract_info(tmp_path)
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    resume_md = None
                else:
                    st.success("Resume processed successfully.")
                st.session_state.resume_md = resume_md
                st.session_state.resume_path = tmp_path
                # Clear previous analysis when a new resume is loaded
                st.session_state.analysis_result = None

        # Persist settings
        st.session_state.model_type = model_type
        st.session_state.temperature = temperature



def analysis_ui():
    st.title(APP_TITLE)
    st.caption("Upload a resume and analyze it using either an Ollama or Transformers backend.")

    if not st.session_state.resume_path:
        st.info("Upload a resume PDF from the sidebar and click 'Load Resume' to begin, then click 'Analyze Resume'.")
        return

    # Analyze controls
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_clicked = st.button("Analyze Resume", type="primary")
    with col2:
        st.write("")

    # Auto-analyze if result already exists
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if analyze_clicked or st.session_state.analysis_result is None:
        try:
            with st.spinner(f"Analyzing resume with {st.session_state.model} model..."):
                response = scan_resume(st.session_state.model, st.session_state.resume_path)
                st.session_state.analysis_result = response
        except Exception as e:
            st.error(f"Error generating analysis: {e}.")
            return

    # Display the result
    st.subheader("Analysis Result")
    st.markdown(st.session_state.analysis_result or "No analysis generated yet.")


def main():
    init_session_state()
    sidebar_controls()
    analysis_ui()


if __name__ == "__main__":
    main()
