import streamlit as st
import os
import PyPDF2 as pdf
import json
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


embeddings = HuggingFaceEmbeddings()


vector_store = FAISS.from_texts(["dummy text"], embedding=embeddings)

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def generate_resume_analysis(resume_text, job_description):
    """Generates resume analysis using a retrieval-based pipeline."""
    
    documents = [
        {"text": resume_text, "metadata": {"source": "resume"}},
        {"text": job_description, "metadata": {"source": "job_description"}}
    ]
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = [text_splitter.split_text(doc["text"]) for doc in documents]
    
    doc_texts = [chunk for sublist in split_docs for chunk in sublist]
    vector_store = FAISS.from_texts(doc_texts, embedding=embeddings)
    retriever = vector_store.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    prompt = f"""
    You are an advanced ATS (Application Tracking System) with expertise in resume analysis.
    Analyze the following resume against the given job description. Provide insights including:
    - Strengths and areas of improvement.
    - Missing skills and keywords.
    - ATS compatibility score (percentage-based).
    - Suggestions for better alignment with the job description.

    Resume:
    {resume_text}
    
    Job Description:
    {job_description}

    Provide a structured JSON response with the following keys:
    {{"ATS Score": "", "Missing Keywords": [], "Profile Summary": "", "Improvement Suggestions": ""}}
    """
    
    response = qa_chain.run(prompt)
    return json.loads(response)

st.title("Smart ATS - Resume Analyzer")
st.text("Analyze your resume and get optimization tips for better ATS compatibility.")

job_description = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume (PDF Only)", type="pdf", help="Please upload a PDF resume.")
submit = st.button("Analyze Resume")

if submit:
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        analysis_result = generate_resume_analysis(resume_text, job_description)
        
        st.subheader("ATS Resume Analysis")
        st.json(analysis_result)
