import streamlit as st
import pandas as pd
from docx import Document
import os

# LLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# Load environment variable for the GROQ API key
groq_api_key = os.getenv("groq_api_key")
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')  # llama3-8b-8192, llama-3.1-70b-versatile

SAVE_DIR = 'documents'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        return ' '.join(df.astype(str).agg(' '.join, axis=1).tolist())
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return ' '.join([para.text for para in doc.paragraphs])
    else:
        st.error("Unsupported file type. Please upload a .txt, .csv, or .docx file.")
        return None

def save_text_to_file(text, filename):
    file_path = os.path.join(SAVE_DIR, filename)
    with open(file_path, 'w') as f:
        f.write(text)
    return file_path

def cleanup_documents():
    """Remove files in the SAVE_DIR after processing."""
    for filename in os.listdir(SAVE_DIR):
        file_path = os.path.join(SAVE_DIR, filename)
        os.remove(file_path)

# Streamlit app layout
st.title("ReviewAI")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "docx"])

if uploaded_file is not None:
    text = extract_text_from_file(uploaded_file)
    st.success("File uploaded successfully!")
    if text:
        # Save the extracted text to a file in the documents directory
        saved_file_path = save_text_to_file(text, uploaded_file.name)

        # Load documents for embeddings creation
        def load_documents():
            directory_path = SAVE_DIR
            loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
            return loader.load()

        # Create the RAG chain
        def create_rag_chain(split_docs):
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embeddings_dir = "chroma_embeddings"  # Directory to store embeddings

            vector_store = Chroma(collection_name="document_embeddings", embedding_function=embedding_model, persist_directory=embeddings_dir)

            # Reset the collection to clear previous embeddings
            vector_store.reset_collection()  # Clear any existing embeddings

            # Add the new documents to the vector store
            vector_store.add_documents(split_docs)

            retriever = vector_store.as_retriever()
            return RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=retriever
            )

        # Initialize and create the RAG chain
        documents = load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        if split_docs:
            rag_chain = create_rag_chain(split_docs)

            # Clean up the saved documents
            cleanup_documents()

            # Create two columns for side-by-side dropdowns
            col1, col2 = st.columns(2)

            # Dropdown for selecting number of questions in the first column
            with col1:
                question_count = st.selectbox("Select the number of questions to generate:", options=[5, 10, 15, 20])

            # Dropdown for selecting question complexity in the second column
            with col2:
                complexity = st.selectbox("Select question complexity:", options=['Easy', 'Medium', 'Hard'])

            # Initialize session state for questions and answers
            if "questions" not in st.session_state:
                st.session_state.questions = []
                st.session_state.answers = ["" for _ in range(question_count)]

            # Query for generating questions
            if st.button("Generate Questions"):
                user_query = f"""You are an advanced question generation model tasked with creating {question_count} unique questions based exclusively on the provided text. Each question should reflect the specified complexity level of {complexity}.\n
                Your output must consist of questions only, with no additional information or commentary.\n
                ***Document Analysis:*** Carefully analyze the provided documents to identify key themes, concepts, and details.\n
                ***Complexity Level:*** Ensure each question aligns with the specified complexity level ({complexity}).\n
                ***Uniqueness:*** Each question must be distinct, ensuring no overlap in content or phrasing among the questions.\n
                ***Question Format:*** Generate only questions, focusing on open-ended formats that encourage deeper thinking and exploration of the text.\n
                ***Strict Adherence:*** All questions must strictly derive from the content of the documents provided, with no external information or assumptions included."""

                response = rag_chain.invoke({"query": user_query})

                # Get the formatted questions
                questions = response['result']

                # Split questions into a list and filter out any empty strings or instructions
                st.session_state.questions = [q.strip() for q in questions.split('\n') if q.strip() and not q.startswith("Here are")]

                # Initialize answers if they were just generated
                if len(st.session_state.answers) != len(st.session_state.questions):
                    st.session_state.answers = [""] * len(st.session_state.questions)
                st.session_state.generated = True
                st.session_state.answers = [""] * len(st.session_state.questions)

            # Display questions and corresponding input fields
            for i, question in enumerate(st.session_state.questions):
                st.markdown(f"**{question}**")  # Display the question
                st.session_state.answers[i] = st.text_input(f"Answer:", value=st.session_state.answers[i], key=question)  # Input for answers

            if st.session_state.questions:
                # Submit button for checking answers with a unique key
                if st.button("Submit Answers", key="submit_answers"):
                    # Construct the LLM query for validation
                    answers_str = "\n".join(f"{q}: {a}" for q, a in zip(st.session_state.questions, st.session_state.answers))
                    validation_query = f"Based on the questions and provided answers, please evaluate the following:\n\nQuestions and Answers:\n{answers_str}\n\nPlease provide feedback on which answers are correct or incorrect."

                    # Call the LLM for answer validation
                    validation_response = rag_chain.invoke({"query": validation_query})
                    st.success(validation_response['result'])  # Display the LLM's feedback

            # Reset button for new questions with a unique key
            if st.session_state.get("generated", False):
                if st.button("Reset", key="reset"):
                    st.session_state.questions = []  # Clear questions
                    st.session_state.answers = [""] * len(st.session_state.questions)
                    st.session_state.generated = False  # Reset the generated state
                    st.success("Questions and answers have been reset!")
