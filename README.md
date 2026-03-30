# RAG Demo — NASA Remote Sensing Q&A

This project implements a dynamic Retrieval-Augmented Generation (RAG) web application using Python, Gradio, FAISS, Sentence Transformers, and OpenAI's API. It allows users to dynamically upload a PDF document and compare a context-constrained AI response with a general, unconstrained AI response side-by-side.

## Project Features

- **Dynamic File Uploads**: Upload any PDF directly through the web UI to build an instant semantic knowledge base.
- **Side-by-Side Dual Outputs**: Easily evaluate how an LLM performs when strictly grounded to a reference document (RAG) versus when answering entirely from its pre-trained general knowledge (Raw LLM).
- **In-Memory Analytics**: The app chunks, embeds (`all-MiniLM-L6-v2`), and indexes (`faiss`) files automatically without saving intermediate files locally.
- **Premium UI**: Uses Gradio's advanced theming engines and custom CSS variables to create a highly aesthetic, minimal, and modern interface.

## Setup Instructions

1. **Virtual Environment**: 
   Ensure you are in the `rag_demo` folder and activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Dependencies**:
   If not already installed, run:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Open the `.env` file in the root folder and ensure you have set your API token. 
   *(Ensure your GitHub token has the **`models`** permission assigned explicitly).*
   ```env
   AI_TOKEN=your_actual_token_here
   ```

## Workflow Walkthrough

1. **Data Ingestion**:
   When you upload a PDF on the web interface, the function `process_uploaded_pdf()` triggers. It extracts text via PyMuPDF inside 500-character segments. These chunks are embedded into dense vectors using HuggingFace sentence transformers, which populate a global memory `faiss` database for fast semantic searches.

2. **RAG vs RAW LLM Evaluation**:
   When a user clicks "Compare Answers":
   - **RAG Flow**: The question queries the FAISS index to find 3 semantically identical context chunks. The system prompt restricts `gpt-4o-mini` unconditionally to only answer using those passages.
   - **RAW Flow**: The identical question bypasses FAISS completely and allows `gpt-4o-mini` to simply utilize the extent of its native knowledge graph.

## Running the Application

To launch the web interface, run:
```powershell
python app.py
```
Then open the local URL (i.e. `http://127.0.0.1:7860`) in your web browser. Try testing out scope restrictions ("What is the capital of France?") to see the RAG block hallucination while the RAW LLM freely answers!
