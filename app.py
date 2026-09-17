import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 1. Page Configuration & Styling ---
st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 IEEE RAS AI Assistant")
st.caption("A RAG-powered assistant providing information on the IEEE Robotics and Automation Society.")

# --- 2. Secure API Key Retrieval ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# If not found in .env, check if we are on the Streamlit Cloud server
if not api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not api_key:
    st.error("GROQ_API_KEY not found. Please verify it is added perfectly inside your local .env file.")
    st.stop()


# --- 3. Initialize RAG Pipeline (Cached for performance) ---
@st.cache_resource
def init_rag_system():
    # Load your existing local vector store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "vector_store", embedding_model, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Initialize the fast cloud LLM
    # Change from "llama3-8b-8192" to "llama-3.1-8b-instant"
    # Change from "llama-3.1-8b-instant" to "openai/gpt-oss-20b"
    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=api_key, temperature=0.2)


    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an official AI Assistant for the IEEE Robotics and Automation Society (RAS). 
        Answer the user's question using ONLY the context provided below. 
        If you don't know the answer from the context, politely state: "I don't have that information in my current IEEE RAS knowledge base."
        
        Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain

try:
    conversational_chain = init_rag_system()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}. Make sure the 'vector_store' folder exists in your directory.")
    st.stop()

# --- 4. Session & Chat History Tracking ---
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

config_history = RunnableWithMessageHistory(
    conversational_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Manage the visual chat bubbles in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw past messages so they don't disappear on refresh
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 5. User Input Window ---
if user_query := st.chat_input("Ask me anything about IEEE RAS..."):
    # Add human message to state and UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # Generate response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = config_history.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": "web_chat_session"}},
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
