import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Setup & Config
load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
st.set_page_config(page_title="Syllabus-GPT", page_icon="🎓")

# 2. Load the Local Database we created in ingest.py
@st.cache_resource
def load_rag_system():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # This loads the 'faiss_index' folder from your MSI laptop
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Initialize Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # 3. Custom Prompt (The "Assessment" Focus)
    template = """
    You are an expert academic assistant for SLIIT students. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, say you don't know.
    Context: {context}
    Question: {question}
    Answer:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

# 4. Streamlit UI
st.title("🎓 Syllabus-GPT")
st.markdown("Query your SLIIT lecture notes and generate study materials.")

if os.path.exists("faiss_index"):
    qa_chain = load_rag_system()
    
    # Chat Input
    user_query = st.text_input("Ask a question about your syllabus:")
    
    if user_query:
        with st.spinner("Analyzing lecture notes..."):
            response = qa_chain.invoke({"query": user_query})
            st.write("### Answer:")
            st.write(response["result"])
            
    # Professional "Assessment" Feature for Pearson
    if st.button("Generate Practice Quiz"):
        with st.spinner("Generating quiz questions..."):
            quiz_query = "Generate 3 multiple-choice questions based on this content for a practice quiz."
            response = qa_chain.invoke({"query": quiz_query})
            st.success("Practice Quiz Generated!")
            st.write(response["result"])
else:
    st.error("No index found! Please run 'python ingest.py' first.")