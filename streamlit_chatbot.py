from openai import OpenAI
import streamlit as st



st.title("Net-Ops bot🤖")

client=OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.write("API Key Loaded:", bool(st.secrets["OPENAI_API_KEY"]))


if "openai_model" not in st.session_state:
    st.session_state["openai_model"]= "text-moderation-latest"


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt:= st.chat_input("How can I Assist you?"):
    with st.chat_message("user",avatar="😎"):
        st.markdown(prompt)

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response =""
        for response in client.responses.create(
            model = st.session_state["openai_model"],
            messages=[
                {"role":m["role"],"content":m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            content_chunk = response.choices[0].delta.content
            if content_chunk is not None:
                full_response += content_chunk
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant","content":full_response})



