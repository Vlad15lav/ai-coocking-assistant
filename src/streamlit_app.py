import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from model.agent import AgentSystem
from tools.downloader import download_from_yandex


@st.cache_resource
def load_agent():
    """Инициализация агента из cache streamlit
    """
    # Инициализация LLM
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-70b-versatile",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0
    )

    # Скачивание FAISS индекса
    faiss_index_path = "data/processed/food_faiss_index"

    if not os.path.exists(faiss_index_path + "index.faiss"):
        download_from_yandex(
            public_link="https://disk.yandex.ru/d/P2qlEw1CgZNMIA",
            save_path=faiss_index_path
            )

    if not os.path.exists(faiss_index_path + "index.pkl"):
        download_from_yandex(
            public_link="https://disk.yandex.ru/d/W-P2GCK64yNufQ",
            save_path=faiss_index_path
            )

    # Инициализация ретривера
    db = FAISS.load_local(
        folder_path="data/processed/food_faiss_index",
        embeddings=HuggingFaceEmbeddings(model_name="sergeyzh/LaBSE-ru-turbo"),
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )

    # Инициализация агента
    agent_executor = AgentSystem(llm=llm, retriever=retriever)

    return agent_executor


# Параметры страницы
st.set_page_config(
        page_title="AI Coocking Assistant | Chat",
        page_icon="🤖",
        layout="wide"
    )

# Инициализация агента
agent_executor = load_agent()

# Левый sidebar
with st.sidebar:
    st.title("🍳 AI Coocking Assistant 🤖")
    st.subheader("Your Chef Assistant")
    st.markdown(
        """
        Welcome to the **AI Coocking Assistant**, your smart kitchen
        helper powered by advanced **Large Language Model**
        technology and **LangChain**🦜🔗.
        This assistant is designed to streamline your cooking process, offering
        tailored recipe suggestions, step-by-step cooking guidance, and helpful
        tips based on your preferences and ingredients on hand.

        Whether you're a seasoned chef or a kitchen beginner, the AI Coocking
        Assistant can make your cooking experience more enjoyable, efficient,
        and creative. Ready to elevate your culinary skills?
        Let's cook together!
        """
    )

st.title("Your Chat")

# Инициализация сессии
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ввод запроса пользователя
audio_input = st.experimental_audio_input("Голосовой запрос")
text_input = st.chat_input("Ваш запрос")

# Логика ChatBot
if text_input or audio_input:
    user_query = text_input

    if user_query:
        # Вызов агента
        agent_result = agent_executor.invoke(user_query)

        # Запрос пользователя
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            })

        # Ответ агента
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": agent_result['output']
            })

# Вывод чата из сессии
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
