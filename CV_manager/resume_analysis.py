from typing import Literal
import gc
from docling.document_converter import DocumentConverter
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

HF_model = "Qwen/Qwen3-4B-Instruct-2507-FP8"
ollama_model_qwen = "qwen3:4b-instruct-2507-q8_0"
ollama_model_gemma3 = "gemma3:4b-it-q4_K_M"
ollama_model_mistral = "mistral:7b-instruct"

def extract_info(pdf_path: str) -> str:
    """
    Extracts content from a PDF file and returns it as a Markdown string.

    This function processes a PDF file using a DocumentConverter, converting its
    contents to Markdown format. It ensures proper cleanup of resources and GPU
    memory after processing. Only files with the ".pdf" extension are supported.

    :param pdf_path: Path to the PDF file to be processed
    :type pdf_path: str
    :return: Content of the PDF file in Markdown format
    :rtype: str
    :raises ValueError: If the provided file path does not end with ".pdf"
    """
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Unsupported file format")

    converter = None
    conv_res = None
    try:
        converter = DocumentConverter()
        conv_res = converter.convert(source=pdf_path)
        pdf_mk = conv_res.document.export_to_markdown()
        return pdf_mk
    finally:
        # Drop references so GPU memory can be reclaimed
        conv_res = None
        converter = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # torch may not be installed or CUDA not present; ignore
            pass


def scan_resume(model:Literal["qwen3:4b_transformers", "qwen3:4b_ollama", "gemma3:4b_ollama", "mistral:7b_ollama"],
                text:str) -> str:
    """
    Processes a given resume text using a specified model to extract insights. Allows for the usage
    of various configured models to analyze the resume content and produce insights based on
    predefined output formats and rules.

    :param model: The specific model to utilize for processing. This can be one of the following:
        - "qwen3:4b_transformers"
        - "qwen3:4b_ollama"
        - "gemma3:4b_ollama"
        - "mistral:7b_ollama"
    :param text: The raw textual content of the resume to be analyzed.
    :return: A string containing generated insights obtained from processing the resume using the
        specified model.
    :raises ValueError: Raised when the `model` parameter does not match any of the supported
        model types.

    """
    pdf_mk = extract_info(text)
    if model == "qwen3:4b_transformers":
        tokenizer = AutoTokenizer.from_pretrained(HF_model)
        model = AutoModelForCausalLM.from_pretrained(
            HF_model,
            dtype="auto",
            device_map="auto",
            temperature=0.7
        )
        prompt = "Provide the insights for this resume:\n\n" + pdf_mk
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=16384
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        result = tokenizer.decode(output_ids, skip_special_tokens=True)

    elif model in ["qwen3:4b_ollama", "gemma3:4b_ollama", "mistral:7b_ollama"]:
        dict_model = {
            "qwen3:4b_ollama": ollama_model_qwen,
            "gemma3:4b_ollama": ollama_model_gemma3,
            "mistral:7b_ollama": ollama_model_mistral,
        }
        model = ChatOllama(model=dict_model[model], temperature=0.7)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that helps with CV's."),
                ("human", "Provide the insights for this resume:\n\n{text}"),
            ]
        )

        text_prompt = prompt.format(text=pdf_mk)

        result = model.invoke(text_prompt).content
    else:
        raise ValueError("Unsupported model type")

    return result
