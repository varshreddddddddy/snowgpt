from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
from streamlit_chat import message
from utils import *
from config import *
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

title = '<p style= "font-family:Calibri; text-align: center; color:purple; font-size: 40px;"><b>SnowGPT Anblicks-Chatbot</b></p>'

st.write(title ,unsafe_allow_html=True)

subheader_style = (
    'font-family: Arial, sans-serif;'
    'font-size: 20px;'
    'color: #2C3FCC  ;'
    'text-align: left;'
)


snowflake_context = "This chatbot is based on Snowflake Documentation"

st.write(f'<p style="{subheader_style}">{snowflake_context}</p>', unsafe_allow_html=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


show_history = st.sidebar.button("Show History")

 

if show_history:

    with st.sidebar.expander("Query History"):

        for i, request in enumerate(st.session_state['requests']):

            st.write(f"{i + 1}. {request}")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.chat_input("Ask me anything in snowflake ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            res=st.chat_message("assistant")
            res.write(st.session_state['responses'][i],key=str(i))
            # message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                req=st.chat_message("user")
                req.write(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                # message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          