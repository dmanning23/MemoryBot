import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, 
                               MessagesPlaceholder, 
                               SystemMessagePromptTemplate,
                               HumanMessagePromptTemplate)
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

@st.cache_resource
def InitializeMemory():
    print("resetting memory")
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def SelectModel():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    temperature = st.sidebar.slider("Temperature:", 
                                    min_value=0.0, 
                                    max_value=2.0,
                                    value=0.0,
                                    step=0.01)
    
    return ChatOpenAI(model=model_name, 
                      openai_api_key=st.secrets["OPENAI_API_KEY"],
                      temperature=temperature)

def InitializeModel(memory):

    llm = SelectModel()

    # Prompt
    prompt = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    return LLMChain(llm=llm, prompt=prompt, memory=memory)

def Run(memory, crc):
    container = st.container()
    with container:
        with st.form(key="my form", clear_on_submit=True):
            user_input  = st.text_area(label="Question: ", key="input", height = 100)
            submit_button = st.form_submit_button(label="Ask")

        if submit_button and user_input:

            with st.spinner("Thinking..."):
                question = {'question': user_input}
                response = crc.run(question)
            
            #write the ressponse
            st.write(response)

            #write the chat history
            variables = memory.load_memory_variables({})
            messages = reversed(variables['chat_history'])
            for message in messages:
                if isinstance(message, AIMessage):
                    with st.chat_message('assistant'):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('user'):
                        st.markdown(message.content)
                else:
                    st.write(f"System message: {message.content}")

def main():

    st.set_page_config(
        page_title="ChatBot With Memory",
        page_icon="🧠")
    
    st.title("🤖 ChatBot with Memory 🧠")
    
    #setup the sidebar
    st.sidebar.title("Options")

    memory = InitializeMemory()

    #add a button to the sidebar to start a new conversation
    clear_button = st.sidebar.button("New Conversation", key="clear")
    if (clear_button):
        print("Clearing memory")
        memory.clear()

    crc = InitializeModel(memory)
    Run(memory, crc)
    
if __name__ == "__main__":
    main()