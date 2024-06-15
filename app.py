import os
import io
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Functions to process PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to process image files
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text += pytesseract.image_to_string(image)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

def ingest_data(uploaded_files=None):
    if uploaded_files:
        raw_text = ""
        
        pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]
        image_files = [f for f in uploaded_files if f.type in ["image/png", "image/jpeg", "image/jpg"]]
        
        if pdf_files:
            pdf_text = get_pdf_text([io.BytesIO(pdf.read()) for pdf in pdf_files])
            raw_text += pdf_text

        if image_files:
            image_text = get_image_text([io.BytesIO(image.read()) for image in image_files])
            raw_text += image_text
        
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks)
        st.success("Files processed successfully!")
    else:
        # Process files from the dataset folder
        pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks)
        st.success("Dataset files processed successfully!")

def get_conversational_chain():
    prompt_template = """
    You are LawMate, a highly experienced attorney providing legal advice based on Indian laws. 
    You will respond to the user's queries by leveraging your legal expertise and the provided information.
    Provide the Section Number for every legal advice.
    Provide Sequential Proceedings for Legal Procedures if to be provided.
    Remember you are an Attorney, so don't provide any other answers that are not related to Law or Legality.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        system_instruction="You are LawMate, a highly experienced attorney providing legal advice based on Indian laws. You will respond to the user's queries by leveraging your legal expertise and the Context Provided.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    qa_chain = get_conversational_chain()
    response = qa_chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)["output_text"]
    return response

def main():
    st.set_page_config("LawMate", page_icon=":scales:")
    st.header("LawMate :scales:")

    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload PDF and Image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if st.sidebar.button("Process Files"):
        ingest_data(uploaded_files)
        
    if not os.path.exists("Faiss"):
        # Process dataset files if Faiss directory doesn't exist
        ingest_data()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm LawMate, an AI Legal Advisor."}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_question = st.chat_input("Type your legal question here:")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = user_input(user_question, chat_history)
                    st.write(response)

            if response is not None:
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
