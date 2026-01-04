import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

import speech_recognition as sr
from gtts import gTTS

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ================== ENV ==================
load_dotenv()

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Chetan Patil Voice Chatbot",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤🤖 Chetan Patil Voice Resume Chatbot")
st.write("Ask questions by **voice or text** based on resume")

# ================== LOAD RESUME ==================
@st.cache_resource
def load_vectorstore():
    if not os.path.exists("tresume.txt"):
        st.error("❌ tresume.txt not found")
        st.stop()

    loader = TextLoader("tresume.txt", encoding="utf-8")
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return FAISS.from_documents(documents, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ================== GEMINI LLM ==================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

# ================== PROMPT ==================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a resume chatbot.

Rules:

-If a person asks another person information please give


Context:
{context}

Question:
{question}

Answer:
"""
)

# ================== VOICE FUNCTIONS ==================
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = r.listen(source)

    try:
        return r.recognize_google(audio)
    except:
        return None


def speak_text(text):
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    st.audio(temp_file.name)

# ================== SESSION STATE ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== TEXT INPUT ==================
text_input = st.chat_input("Type your question...")

if text_input:
    docs = retriever.invoke(text_input)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context,
        question=text_input
    )

    response = llm.invoke(final_prompt).content

    st.session_state.messages.append(("user", text_input))
    st.session_state.messages.append(("assistant", response))

    speak_text(response)

# ================== VOICE INPUT ==================
if st.button("🎤 Ask by Voice"):
    voice_text = get_voice_input()

    if voice_text:
        st.session_state.messages.append(("user", voice_text))

        docs = retriever.invoke(voice_text)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.format(
            context=context,
            question=voice_text
        )

        response = llm.invoke(final_prompt).content

        st.session_state.messages.append(("assistant", response))

        speak_text(response)
    else:
        st.error("❌ Could not understand voice")

# ================== CHAT DISPLAY ==================
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)
