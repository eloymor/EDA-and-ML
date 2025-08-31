# CV Manager — Resume Creation and Analysis
This project provides two Streamlit apps and supporting modules to help you:
- Create a polished resume from your own inputs using a local LLM.
- Analyze an existing resume PDF and get insights using either an Ollama-backed model or a Transformers model.

The core components:
- app_cv_creator.py: Streamlit UI to generate resumes.
- resume_creator.py: Resume generation logic using a local LLM via LangChain + Ollama.
- app_cv_analysis.py: Streamlit UI to upload and analyze resume PDFs.
- resume_analysis.py: PDF-to-Markdown extraction and model-backed analysis.

## Features
- Resume creation UI: enter your details and resume text; get a formatted resume (Markdown) with a one-click download.
- Resume analysis UI: upload a PDF, convert to Markdown, and analyze it for insights.
- Model flexibility:
    - Local Ollama models (e.g., Qwen, Gemma, Mistral).
    - Hugging Face Transformers (Qwen) with automatic device mapping.

- Environment-tunable behavior via environment variables.

## Prerequisites
- Python 3.13.5
- uv (project package manager)
- Optional but recommended:
    - GPU and CUDA if you plan to use Transformers for larger models
    - [Ollama](https://ollama.com) installed and running for local LLMs

## Install dependencies
This project uses uv for dependency management. From the project root:
- Install project dependencies declared in pyproject.toml:
    ```bash
    uv sync
    ```
- If you need to add missing libraries for these apps (examples):
    ```bash
    uv add streamlit langchain-ollama docling transformers torch
    ```
Notes:
- torch installation may vary per platform; if uv fails to resolve a suitable build, follow PyTorch’s official installation instructions, then run uv sync.
- If you plan to only use Ollama, torch/transformers are not strictly required.

## Model setup (Ollama)
Ensure Ollama is running, then pull the models you plan to use. For example:
```bash
    ollama pull qwen3:4b-instruct-2507-q8_0
    ollama pull gemma3:4b-it-q4_K_M
    ollama pull mistral:7b-instruct
 ```

You can customize which Ollama model to use via the UI in the analyzer app or environment variables.
## How to run
You can run the apps from the project root using uv:
- Resume Creator:
    ```bash
    uv run streamlit run CV_manager/app_cv_creator.py
    ```
- Resume Analyzer:
    ```bash
    uv run streamlit run CV_manager/app_cv_analysis.py
    ```
If you prefer running without uv (assuming your environment is already set up):
```bash
    streamlit run CV_manager/app_cv_creator.py
    streamlit run CV_manager/app_cv_analysis.py
   ```

## Typical workflows
- Create a resume
    1. Start the Resume Creator app.
    2. Fill out name, position, contact info, and paste your resume content/notes.
    3. Click Create Resume to generate Markdown output.
    4. Preview and download as a .md file.

- Analyze a resume
    1. Start the Resume Analyzer app.
    2. In the sidebar, choose your model and upload a PDF.
    3. Click Load Resume to parse it to Markdown.
    4. Click Analyze Resume to generate insights.

## Troubleshooting
- Ollama connection errors:
    - Ensure the Ollama service is running.
    - Verify the model name exists locally (ollama list) and that you pulled it.

- PDF extraction issues:
    - Only .pdf files are supported.
    - Ensure docling is installed and compatible with your OS.

- Transformers resource errors:
    - Large models may require significant RAM/GPU memory.
    - Set a smaller model or switch to Ollama in the analyzer UI.

- Torch/CUDA problems:
    - If CUDA isn’t available, Transformers will run on CPU and may be slow.
    - Install a CUDA-enabled build of PyTorch if applicable for your system.

## Privacy and security
- Uploaded PDFs are stored locally in a temporary directory for processing.
- No data is sent to third-party services when using local Ollama or local Transformers models.
- Remove temporary files after use if you’re processing sensitive data.

## Project layout (relevant files)
- CV_manager/app_cv_creator.py — Streamlit app for resume creation
- CV_manager/app_cv_analysis.py — Streamlit app for resume analysis
- CV_manager/resume_creator.py — Resume generation logic
- CV_manager/resume_analysis.py — PDF extraction and analysis logic

## License
MIT license.
