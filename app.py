import subprocess
import sys

# Ensure installation inside Streamlit's virtual environment
#subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "pymupdf"])

 # pymupdf for PDF text extraction


# Manually add user site-packages to sys.path
#import site
#sys.path.append(site.getusersitepackages())

# pymupdf for PDF text extraction


import streamlit as st
  # pymupdf for PDF text extraction



  # pymupdf for PDF text extraction
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Replace with your actual Groq API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n\n".join([page.get_text("text") for page in doc])
    return text

# Function to generate text embeddings
def get_embeddings(text_chunks):
    return model.encode(text_chunks, convert_to_tensor=False)

# Function to create a FAISS index
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Function to retrieve relevant text using FAISS
def retrieve_similar_text(query, text_chunks, index):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top-3 chunks
    retrieved_texts = " ".join([text_chunks[i] for i in indices[0]])
    return retrieved_texts

# Function to generate answers using Groq's LLM
def generate_answer(context, query):
    system_prompt = f"""
    You are an AI assistant. Use the following context to answer the user's question.
    Context: {context}
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=1.0,
        stream=False,
    )
    return response.choices[0].message["content"]

# Streamlit UI
st.title("📄 Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text and split into chunks
        text = extract_text_from_pdf(uploaded_file)
        chunks = text.split("\n\n")

        # Generate embeddings and create FAISS index
        embeddings = get_embeddings(chunks)
        index = create_faiss_index(np.array(embeddings))

    st.success("PDF processed! You can now ask questions.")

    query = st.text_input("❓ Ask a question:")

    if query:
        with st.spinner("Fetching answer..."):
            context = retrieve_similar_text(query, chunks, index)
            answer = generate_answer(context, query)
        st.write("💡 Answer:", answer)
