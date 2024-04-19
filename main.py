import streamlit as st
import boto3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title='Start Consulting', layout='wide', page_icon="./fav.ico")
st.title("Chat with Anastasia")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

def data_ingestion():
    loader = PyPDFDirectoryLoader("database")
    data = loader.load()
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
                                                chunk_size=1000,
                                                chunk_overlap=200)
    docs = data_split.split_documents(data)
    return docs

def vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def claude_model():
    llm = BedrockChat(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={
            "max_tokens": 2048,
            "temperature": 0.9,
            "top_k": 250,
            "top_p": 1})
    return llm

prompt_template = """
You're an AI expert in psychology and psychiatry named Anastasia.
You have several task is to give mental health and personal development advice to users and become their mental support. You will be confused if the user ask about something out of topic. Only give the user about the professional data when only they ask. Be smart and creative. don't throw the same answer in every response. 

<context>
{context} 
</context>

Question: {question}
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

if "memory" not in st.session_state:
    st.session_state.memory = StreamlitChatMessageHistory()

# Sidebar info
with st.sidebar:
    st.image('./fav.ico', width=100, use_column_width= "never")
    st.write("## Anastasia.ai")
    st.caption("Personal AI Assistant in Mental Health and Personal Development")

    st.divider()
    st.markdown("If you want to ask about our mental health professional's information. Make sure to click 'Sync database'first")
    if st.button("Sync database"):
        with st.spinner("Processing..."):
            docs = data_ingestion()
            vector_store(docs)
            st.success("Done")
    
# Conversation
for message in st.session_state.memory.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input
user_input = st.chat_input()
if user_input:
    human_message = HumanMessage(content=user_input)
    st.session_state.memory.add_user_message(human_message)
    with st.chat_message("Human"):
        st.markdown(user_input)
    with st.chat_message("AI"):
        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = claude_model()
        response = get_response_llm(llm, faiss_index, user_input)
        ai_message = AIMessage(content=response)
        st.write(response)
        st.session_state.memory.add_ai_message(ai_message)
