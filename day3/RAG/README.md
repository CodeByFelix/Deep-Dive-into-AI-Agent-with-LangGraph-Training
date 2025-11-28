# RAG Demo (Simple)

This demo shows a minimal Retrieval-Augmented Generation pipeline using **sentence-transformers** + **FAISS** for retrieval, and **OpenAI** for generation. It's designed for teaching and local experimentation.

## Files
- `demo_rag.py` — Streamlit app for interactive demo
- `rag_utils.py` — Simple RAG helper (builds FAISS index, queries)
- `requirements.txt` — Python dependencies

## Quick start
1. Create a Python virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key (optional, for generation):
   - `export OPENAI_API_KEY='your_key_here'` (Linux/macOS)
   - `setx OPENAI_API_KEY "your_key_here"` (Windows)
4. Run the Streamlit app:
   - `streamlit run demo_rag.py`
5. In the app: choose sample docs or upload text files, enter a question, and see retrieved context + generated answer.

## Notes for teaching
- Explain chunking and indexing.
- Show retrieval scores and discuss prompt design.
- Swap out OpenAI generation for any LLM (e.g., local Llama) by changing the generation block.
