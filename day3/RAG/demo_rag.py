import os
import streamlit as st
from rag_utils import SimpleRAG
from clarifai.client.model import Model
import dotenv

dotenv.load_dotenv()

st.set_page_config(page_title='RAG Demo', layout='wide')

st.title('Retrieval-Augmented Generation (RAG) — Live Demo')
st.markdown('Simple demo using **sentence-transformers** + **FAISS** for retrieval and **OpenAI** for generation.\n\nSet environment variable `OPENAI_API_KEY` before running.')

# Sidebar: Upload docs or use sample
st.sidebar.header('Data Source')
data_choice = st.sidebar.radio('Choose documents', ('Use sample docs', 'Upload .txt files'))

model_url = "https://clarifai.com/openai/chat-completion/models/GPT-4"
docs = []
if data_choice == 'Use sample docs':
    # sample texts
    docs = [
        'RAG (Retrieval-Augmented Generation) combines a retrieval component with a generative model to produce up-to-date and factual answers. The retriever finds relevant documents, and the generator conditions on them to craft responses.',
        'FAISS is a library for nearest neighbor search that allows efficient similarity search over dense vectors. It works well with sentence-transformer embeddings.',
        'Chunking strategy affects retrieval performance: sentences, paragraphs, or fixed-size windows can be used. Good chunking balances semantic completeness and size.'
    ]
else:
    uploaded = st.sidebar.file_uploader('Upload one or more .txt files', type=['txt'], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            docs.append(f.read().decode('utf-8'))

st.sidebar.markdown('---')
top_k = st.sidebar.slider('Top K retrieved documents', 1, 5, 3)

# Build or load index
if 'rag' not in st.session_state:
    if docs:
        st.session_state['rag'] = SimpleRAG()
        with st.spinner('Building index...'):
            st.session_state['rag'].build_index(docs)
        st.success('Index built with %d documents' % len(docs))
    else:
        st.info('Please select sample docs or upload .txt files.')

# Query UI
query = st.text_input('Enter your question here:', '')

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Retrieved context')
    if query and 'rag' in st.session_state:
        results = st.session_state['rag'].query(query, top_k=top_k)
        for idx, score, text in results:
            st.markdown(f'**Doc {idx}** — score: `{score:.4f}`')
            st.write(text)
    else:
        st.write('No query yet.')

with col2:
    st.subheader('Generated answer (LLM)')
    if query:
        if 'CLARIFAI_API_KEY' not in os.environ:
            st.warning('Set CLARIFAI_API_KEY in environment to enable generation.')
        else:
            # Prepare prompt with retrieved context
            results = st.session_state['rag'].query(query, top_k=top_k)
            context = '\n\n'.join([r[2] for r in results])
            prompt = f"""You are an assistant that answers questions using the provided context.\nContext: {context}\n---\nQuestion: {query}\nAnswer concisely and cite which context doc (by Doc #) you used in your answer."""
            with st.spinner('Generating answer from OpenAI...'):
                # response = openai.ChatCompletion.create(
                #     model='gpt-3.5-turbo',
                #     messages=[{'role':'user','content':prompt}],
                #     max_tokens=300,
                #     temperature=0.2
                # )
                # answer = response.choices[0].message.content.strip()
                inference_params = dict(temperature=0.5, max_tokens=1024)
                model_output = Model(
                    url=model_url,pat=os.getenv("CLARIFAI_API_KEY")).predict_by_bytes(prompt.encode(),
                    input_type="text", 
                    inference_params=inference_params
                )
                response = model_output.outputs[0].data.text.raw
                st.write(response)
    else:
        st.write('Type a question and press Enter to run the demo.')

st.markdown('---')
st.markdown('**Notes:** This demo is intentionally simple for teaching. In production you would: 1) chunk documents, 2) store embeddings in a persistent vector store, and 3) handle prompt engineering and safety checks.')
