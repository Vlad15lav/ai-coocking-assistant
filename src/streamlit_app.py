import os
import logging
import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from model.agent import AgentSystem
from tools.downloader import download_from_yandex
from tools.utils import spech2text


@st.cache_resource
def load_agent():
    """Инициализация агента из cache streamlit
    """
    # Инициализация LLM
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
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
    agent_executor = AgentSystem(llm=llm, retriever=retriever, k=6)

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
    st.subheader("Your Smart Kitchen Guide")
    st.markdown(
        """
        Добро пожаловать в **AI Cooking Assistant**
        — вашего интеллектуального кухонного помощника,
        работающего на основе искусственного интеллекта,
        который помогает находить идеальные рецепты и
        удивлять уникальными кулинарными решениями!

        Этот высокотехнологичный ассистент использует передовые технологии
        **Large Language Model** 🤖 и фреймворк **LangChain** 🦜🔗,
        чтобы предложить персонализированные рекомендации по рецептам,
        создать уникальные блюда и предоставить возможность
        визуального сравнения различных кулинарных решений.

        🍽️ Начните общение с нашим AI помощником:\n
        🥕 Попросите порекомендовать рецепт на основе ваших ингредиентов.<br/>
        ✨ Попросите создать новый рецепт на основе ваших ингредиентов.<br/>
        📸 Попросите изображение блюда для вдохновения или ваших задач.<br/>

        Приятного приготовления! 👨‍🍳👩‍🍳
        """,
        unsafe_allow_html=True)

    # Голосовой ввод
    audio_input = st.experimental_audio_input("Голосовой ввод 🎙️")

st.title("🤖AI ChatBot💬 (Online 🟢)")

# Инициализация сессии и памяти агента
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []

agent_executor.initial_memory(st.session_state.agent_memory)

# Ввод запроса пользователя
text_input = st.chat_input("Какие ингредиенты предпочитаете?")

# Вывод чата из сессии
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            image_content = message["content"]

            st.image(
                image=image_content['image'],
                caption=image_content['caption']
                )

# Логика ChatBot
if text_input or audio_input:
    user_query = text_input
    if text_input is None:
        speech_text = asyncio.run(spech2text(audio_file=audio_input))
        user_query = speech_text['text'].strip()

    # Запрос пользователя
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        })

    # Вызов агента
    try:
        agent_result = agent_executor.invoke(user_query)
        st.session_state.agent_memory = agent_executor.get_chat_history()
    except Exception as e:
        agent_result = {
            "output": None
            }
        logging.error(e)

    # Ответ агента
    with st.chat_message("assistant"):
        if isinstance(agent_result['output'], str):
            agent_content = agent_result['output']

            st.markdown(agent_result['output'])
        elif agent_result['output'] is None:
            agent_content = "Не могу обработать запрос, " + \
                "попробуйте чуть позже!😓"

            st.markdown(agent_content)
        else:
            agent_content = {
                "image": agent_result['output'],
                "caption": "Источник: " + agent_result['url']
            }

            st.image(
                image=agent_content['image'],
                caption=agent_content['caption']
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": agent_content
        })



