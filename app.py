import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- ENVIRONMENT SETUP ---
load_dotenv()

hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
groq_api_key = os.getenv("GROQ_API_KEY")

if not hf_token or not groq_api_key:
    st.error("Missing API tokens. Set HUGGINGFACEHUB_API_TOKEN and GROQ_API_KEY.")
    st.stop()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GenieMark - Pakistani Marketing Genie",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem; }
        .message { padding: 0.5rem 1rem; border-radius: 0.75rem; margin: 0.5rem 0; }
        .message.user { background-color: #2f3136; color: white; }
        .message.assistant { background-color: #23262d; color: #d4d4d4; }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: #00B386;'>GenieMark - Your Pakistani Marketing Genie</h1>", unsafe_allow_html=True)

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🤖 GenieMark")
    st.caption("AI assistant for Pakistani marketers.")
    st.markdown("#### 📌 Questions Asked:")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"- {msg['content']}")
    st.markdown("----")
    st.markdown("🟢 **Status:** Online")

# --- LOAD RAG CHAIN ---
@st.cache_resource(show_spinner=False)
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
    vector_store = FAISS.load_local("faiss_digital_marketing_knowledgebase", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=512
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a smart and helpful AI assistant working for a top marketing agency in Pakistan.

Your tasks include:
- Sharing latest marketing and advertising trends (Pakistan-specific)
- Answering FAQs about services, pricing, processes
- Assisting users in booking demos
- Recommending suitable services
- Helping generate leads
- Deny responses for unrelated non-marketing queries

Use a friendly, concise tone. Mention local context. Suggest actions like 'Book a demo' when appropriate.

{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    memory_store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

conversational_rag_chain = load_chain()

# --- MAIN CHAT INTERFACE ---
query = st.chat_input("Ask your marketing question (Pakistan-focused only)...")

# --- RENDER FULL MESSAGE HISTORY FIRST ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HANDLE NEW USER QUERY ---
if query:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Display a placeholder for assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("**Thinking...**")

    # Get assistant response
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    answer = result["answer"]

    # Replace placeholder with actual answer
    response_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
