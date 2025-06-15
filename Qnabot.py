import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set in the environment!")   # Load the .env file in the environment
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key


@st.cache_resource
def load_main_vectorstore():                                    # Load the main(fixed) vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("my_vectoredb", embeddings, allow_dangerous_deserialization=True)

main_vectorstore = load_main_vectorstore()

st.title("Hybrid RAG: Ask Across My PDF Database + Your File")  # Title making using streamlit fxn
uploaded_file = st.file_uploader("Upload your PDF (optional):", type=["pdf"])
user_vectorstore = None                                         # This gets the input pdf from the user

if uploaded_file:
    with open("temp.pdf", "rb") as f:
        f.write(uploaded_file.getbuffer())                      # Code for streamlit ui for user file upload 
    loader = PyPDFLoader("temp.pdf")
    user_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    user_splits = splitter.split_documents(user_docs)           # Splits, embedds and stores in vector database(temp)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    user_vectorstore = FAISS.from_documents(user_splits, embeddings)

user_question = st.text_input("Ask a question (searches both databases):")
                                                                # User query input
if user_question:                                               # Similarity search in both the databases
    main_docs = main_vectorstore.similarity_search(user_question, k=3)
    user_docs = user_vectorstore.similarity_search(user_question, k=3) if user_vectorstore else []
                                                                # Combine the context from the similarity search 
    combined_context = "\n\n".join([doc.page_content for doc in main_docs + user_docs])

    prompt_template = """You are a helpful assistant.You are supposed to answer the query in very much detailed oriented approach.The answer should mention the source first and then explain the topic in the query in 50 to 60 sentences.  Answer based on this context:
    {context}
    
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)
                                                                # Augment the user query and the context to LLM and generate output text                                                              
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    formatted_prompt = prompt.format(context=combined_context, question=user_question)
    
    with st.spinner("Thinking..."):
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else response
    
    st.markdown("**Answer:**")
    st.write(answer)
