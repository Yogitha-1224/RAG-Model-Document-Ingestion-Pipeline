import os
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

app = Flask(__name__)
UPLOAD_FOLDER = 'data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set your Google API Key (Get one at aistudio.google.com)
os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_API_KEY_HERE"

# Global variable to hold the vector database in memory
vector_db = None

def get_loader(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf": return PyPDFLoader(file_path)
    if ext == ".docx": return Docx2txtLoader(file_path)
    return TextLoader(file_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest_document():
    global vector_db
    file = request.files['file']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        
        # Load, Chunk, and Embed (Milestone 1 & 2)
        loader = get_loader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return jsonify({"status": "Success", "message": f"Ingested {file.filename}"})

@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_db
    if not vector_db:
        return jsonify({"error": "Please upload a document first."})
    
    query = request.json.get("question")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Retrieval-Augmented Generation (Milestone 3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    
    response = qa_chain.invoke(query)
    return jsonify({"answer": response["result"]})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)