import os
# Add this line to resolve OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import faiss
import torch
import numpy as np
import traceback
from transformers import DistilBertModel, DistilBertTokenizer
from werkzeug.utils import secure_filename
import psutil  # For memory usage logging

app = Flask(__name__)

# Configure CORS to allow all origins for development purposes
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables to store the index, file chunks map, and all document chunks
index = None
file_chunks_map = {}
all_chunks = []

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Function to check if memory usage exceeds a threshold
def check_memory_limit(threshold_mb=1024):  # Set a memory limit in MB
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 ** 2
    if mem_usage > threshold_mb:
        raise MemoryError(f"Memory usage exceeded threshold: {mem_usage:.2f} MB")

# Load pre-trained DistilBERT model from Hugging Face
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()  # Ensure the model is in evaluation mode
    torch.set_num_threads(1)  # Limit the number of threads to avoid over-parallelization
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        traceback.print_exc()
    return text

# Function to extract text from TXT
def extract_text_from_txt(txt_file):
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        traceback.print_exc()
        return ""

# Function to split document into chunks
def split_into_chunks(document_text, chunk_size=150):  # Reduced chunk size
    sentences = document_text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = len(sentence)
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    return chunks

# Function to process chunks in batches
def process_chunks_in_batches(chunks, batch_size=16):  # Adjust batch size
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        inputs = tokenizer(batch_chunks, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():  # Disable gradients for efficiency
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # Move to CPU to free GPU memory
        embeddings.append(batch_embeddings)
        log_memory_usage()  # Log memory after each batch
    return np.vstack(embeddings)

# Route to upload files and create FAISS index
@app.route('/upload', methods=['POST'])
def upload_files():
    global index, file_chunks_map, all_chunks

    try:
        print("Received upload request")
        log_memory_usage()  # Log memory usage at the start of the upload
        uploaded_files = request.files.getlist("files")
        print(f"Number of files: {len(uploaded_files)}")
        
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400

        # Reset global variables
        all_chunks = []
        file_chunks_map = {}

        for uploaded_file in uploaded_files:
            file_name = secure_filename(uploaded_file.filename)
            print(f"Processing file: {file_name}")
            
            if file_name.lower().endswith(".pdf"):
                file_content = extract_text_from_pdf(uploaded_file)
            elif file_name.lower().endswith(".txt"):
                file_content = extract_text_from_txt(uploaded_file)
            else:
                return jsonify({"error": f"Unsupported file format: {file_name}"}), 400

            # Split the document content into chunks
            document_chunks = split_into_chunks(file_content)
            all_chunks.extend(document_chunks)
            file_chunks_map[file_name] = document_chunks

        # Generate embeddings for all chunks using DistilBERT model
        if model is None:
            return jsonify({"error": "Model not loaded properly"}), 500

        log_memory_usage()  # Log memory before generating embeddings

        # Process chunks in batches
        try:
            check_memory_limit()  # Check memory before generating embeddings
            chunk_embeddings = process_chunks_in_batches(all_chunks, batch_size=16)
        except MemoryError as mem_err:
            print(f"Memory error: {mem_err}")
            return jsonify({"error": "Memory limit exceeded during embedding generation"}), 500

        # Create FAISS index for similarity search
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings.astype('float32'))

        log_memory_usage()  # Log memory after generating embeddings and creating FAISS index

        return jsonify({"message": "Files uploaded successfully", "file_map": file_chunks_map})

    except Exception as e:
        print(f"Error in upload_files: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Route to search through uploaded documents
@app.route('/search', methods=['POST'])
def search():
    global index, file_chunks_map, all_chunks

    if not index or not all_chunks:
        return jsonify({"error": "No documents uploaded yet"}), 400

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Encode the query and search for top results
        inputs = tokenizer([query], return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

        k = 3  # Top 3 results
        distances, indices = index.search(query_embedding.astype('float32'), k)

        results = []
        for idx in indices[0]:
            for file_name, chunks in file_chunks_map.items():
                if all_chunks[idx] in chunks:
                    result = {
                        "file_name": file_name,
                        "content": all_chunks[idx][:200]  # Display the first 200 characters
                    }
                    results.append(result)
                    break

        return jsonify({"results": results})
    except Exception as e:
        print(f"Error in search: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
