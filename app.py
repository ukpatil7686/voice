import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from gtts import gTTS
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ================== ENV ==================
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY missing. Add it in .env or Streamlit Secrets")
    st.stop()

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Chetan Patil Resume Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Resume Chatbot (Text + Voice Output)")
st.write("Ask questions based on resume (Text input only)")

# ================== LOAD RESUME ==================
@st.cache_resource
def load_vectorstore():
    if not os.path.exists("tresume.txt"):
        st.error("❌ tresume.txt file not found")
        st.stop()

    loader = TextLoader("tresume.txt", encoding="utf-8")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ================== GEMINI MODEL ==================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

# ================== PROMPT ==================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional resume chatbot.

Rules:
-If a person asks another person information please give


Context:
{context}

Question:
{question}

Answer:
"""
)

# ================== SESSION STATE ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== CHAT INPUT ==================
user_input = st.chat_input("Type your question here...")

if user_input:
    docs = retriever.invoke(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context,
        question=user_input
    )

    response = llm.invoke(final_prompt).content

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))

    # ===== TEXT TO SPEECH =====
    tts = gTTS(text=response, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    st.audio(temp_audio.name)

# ================== CHAT DISPLAY ==================
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)
