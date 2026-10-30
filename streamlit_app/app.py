import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from TXT file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")


# Function to split document into smaller chunks
def split_into_chunks(document_text, chunk_size=300):
    sentences = document_text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Load the pre-trained model
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model



model = load_model()

# Streamlit UI
st.title("Semantic Search Application")

# Upload files
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=['pdf', 'txt'], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    file_chunks_map = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.write(f"Processing {file_name}...")

        # Extract text from the uploaded file based on its format
        if file_name.endswith(".pdf"):
            file_content = extract_text_from_pdf(uploaded_file)
        elif file_name.endswith(".txt"):
            file_content = extract_text_from_txt(uploaded_file)

        # Split the content into smaller chunks
        document_chunks = split_into_chunks(file_content)
        all_chunks.extend(document_chunks)
        file_chunks_map[file_name] = document_chunks

    # Generate embeddings for all chunks
    chunk_embeddings = model.encode(all_chunks)

    # Create FAISS index for similarity search
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    # Search query input
    query = st.text_input("Enter your search query:")

    if query:
        query_embedding = model.encode([query])
        k = 3  # Top 3 results
        distances, indices = index.search(np.array(query_embedding), k)

        # Display the top 3 relevant chunks
        st.write("Top 3 relevant results:")
        for i, idx in enumerate(indices[0]):
            # Find the corresponding file and chunk
            for file_name, chunks in file_chunks_map.items():
                if all_chunks[idx] in chunks:
                    st.write(f"Result {i + 1} from file: {file_name}")
                    st.write(f"Relevant content: {all_chunks[idx][:200]}...")  # Display first 200 characters
                    break