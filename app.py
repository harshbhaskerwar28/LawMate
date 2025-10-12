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
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables!")
    st.stop()

genai.configure(api_key=api_key)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Functions to process PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Error reading PDF: {str(e)}")
    return text

# Function to process image files
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        try:
            image = Image.open(image_file)
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text:
                text += extracted_text
        except Exception as e:
            st.warning(f"Error processing image: {str(e)}")
    return text

def get_text_chunks(text):
    # Reduced chunk size to avoid API limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # Reduced from 50000
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Process chunks in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            try:
                if i > 0:
                    time.sleep(1)  # Add delay between batches
                vector_store_batch = FAISS.from_texts(batch, embedding=embeddings)
                all_embeddings.append(vector_store_batch)
            except Exception as e:
                st.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                time.sleep(2)  # Wait longer on error
                continue
        
        if not all_embeddings:
            raise Exception("Failed to create any embeddings")
        
        # Merge all vector stores
        vector_store = all_embeddings[0]
        for vs in all_embeddings[1:]:
            vector_store.merge_from(vs)
        
        vector_store.save_local("Faiss")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def ingest_data(uploaded_files=None):
    try:
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
            
            if not raw_text.strip():
                st.warning("No text extracted from uploaded files.")
                return False
            
            text_chunks = get_text_chunks(raw_text)
            if create_vector_store(text_chunks):
                st.success("Files processed successfully!")
                return True
            return False
        else:
            # Process files from the dataset folder
            dataset_path = "dataset"
            if not os.path.exists(dataset_path):
                st.info("No dataset folder found. Please upload files to proceed.")
                return False
            
            pdf_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(".pdf")]
            
            if not pdf_files:
                st.info("No PDF files found in dataset folder.")
                return False
            
            raw_text = get_pdf_text(pdf_files)
            if not raw_text.strip():
                st.warning("No text extracted from dataset files.")
                return False
            
            text_chunks = get_text_chunks(raw_text)
            if create_vector_store(text_chunks):
                st.success("Dataset files processed successfully!")
                return True
            return False
    except Exception as e:
        st.error(f"Error in data ingestion: {str(e)}")
        return False

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
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        qa_chain = get_conversational_chain()
        response = qa_chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)["output_text"]
        return response
    except Exception as e:
        return f"Error processing your question: {str(e)}. Please try again or rephrase your question."

def main():
    st.set_page_config("LawMate", page_icon=":scales:")
    st.header("LawMate :scales:")

    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload PDF and Image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if st.sidebar.button("Process Files"):
        with st.spinner("Processing files..."):
            ingest_data(uploaded_files)
    
    # Check if vector store exists, if not try to process dataset
    if not os.path.exists("Faiss"):
        st.info("Initializing vector store from dataset...")
        with st.spinner("Processing dataset files..."):
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
        # Check if vector store exists
        if not os.path.exists("Faiss"):
            st.warning("Please upload and process files first before asking questions.")
        else:
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
