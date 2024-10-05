import streamlit as st


with st.sidebar:
    st.title("🍳 AI Food Assistant 🤖")
    st.subheader("Your Chef Assistant")
    st.markdown(
        """
        Welcome to the **AI Food Assistant**, your smart kitchen helper powered
        by advanced **Large Language Model**
        technology and **LangChain**🦜🔗.
        This assistant is designed to streamline your cooking process, offering
        tailored recipe suggestions, step-by-step cooking guidance, and helpful
        tips based on your preferences and ingredients on hand.

        Whether you're a seasoned chef or a kitchen beginner, the AI Food
        Assistant can make your cooking experience more enjoyable, efficient,
        and creative. Ready to elevate your culinary skills?
        Let's cook together!
        """
    )

st.title("Chef Bot 👨‍🍳")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
        )
