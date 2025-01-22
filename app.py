import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain.chains import RetrievalQA
from io import BytesIO

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDF file
def extract_file(upload_file):
    data = ""  # Start with an empty string to accumulate text

    upload_file.seek(0)  # Reset the file pointer if needed
    pdf_reader = PdfReader(upload_file)  # Pass BytesIO object directly to PdfReader
    
    for page in pdf_reader.pages:
        text = page.extract_text()  # Extract text from each page
        if text:  # Only include non-empty pages
            data += text  # Concatenate the extracted text

    return data


# Split the extracted text into smaller chunks
def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs_data = text_splitter.split_text(data)  # Now, split the plain string into smaller chunks

    return docs_data


# Vectorize the document chunks and create a retriever
def vector_store(docs_data):

    vectorst = Chroma.from_texts(
        texts=docs_data,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="data/chroma_store"  # Specify a persistence directory
    )

    retriever = vectorst.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    return retriever

# Set up the QA chain with a prompt template
def convo_chain(question, docs_data):
    prompt_template = """
    Answer the question with clarity and details from the provided context. If the answer is not in the context,
    say, "answer is not found in the file". Do not give incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=500)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the conversational retrieval chain
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    retriever = vector_store(docs_data)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.invoke(question)
    st.markdown(f"### Reply:\n{answer}")


def main():
    st.title("RAG ChatBot Application")
    st.write("Bring your documents to life!")

    # Check if the file uploader has been used and manage the file upload state
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False

    # File uploader
    uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
    successful_file = False

    if uploaded_file:
        st.success("File successfully loaded")
        successful_file = True
        with st.spinner("Processing your files..."):
            data = extract_file(uploaded_file)

            st.write("Document uploaded successfully!")

            # Split the extracted content into smaller chunks
            docs_data = split_text(data)

    if successful_file:
        user_question = st.text_input("Enter your question:", placeholder="What do you want to know?")
        if user_question:
            with st.spinner("Fetching your answer..."):
                convo_chain(user_question, docs_data)

if __name__ == "__main__":
    main()
