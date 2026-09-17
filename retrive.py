from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Vector store / retriever ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "vector_store", embedding_model, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# --- LLM ---
llm = ChatGroq(
    model="llama3-8b-8192", 
    groq_api_key=api_key, 
    temperature=0.2
)

# --- Prompt (with chat history support) ---
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


# --- RAG chain ---
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

# --- Message history wiring ---
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

session_id = "user_001"

print("Connected to Groq Cloud API successfully!")
print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = conversational_chain.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Bot: {response}\n")