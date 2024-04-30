import os
import tempfile
import streamlit as st
from streamlit_chat import message
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

logging.basicConfig(level=logging.INFO)


class ChatPDFAssistant:
    
    def __init__(self):
        self.model = ChatOllama(model="mistral", port=12345)  # Use port 12345 instead of the default port 11434
        #self.model = ChatOllama(model="mistral")  
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt_with_pdf = self._create_prompt_template_with_pdf()
        self.prompt_without_pdf = self._create_prompt_template_without_pdf()
        self.vector_store = None
        self.retriever = None
        self.chain_with_pdf = None
        self.chain_without_pdf = self._prepare_chain_without_pdf()

    @staticmethod
    def _create_prompt_template_with_pdf():
        return ChatPromptTemplate.from_template(
            """
            <s> [INST] 
            You are an AI assistant with access to specific text snippets for answering questions.
            Your answers must be directly based only on the provided context.
            If the context does not contain enough information for a definitive answer, say that you don't know and ask the user to provide additional information. 
            [/INST] </s> 
            [INST] 
            Question: {question} 
            Context: {context} 
            Answer: 
            [/INST]
            """
        )

    @staticmethod
    def _create_prompt_template_without_pdf():
        return ChatPromptTemplate.from_template(
            """
            <s> [INST] 
            You are an injection molding expert, experienced in giving answers to questions related to the domain of injection molding only. 
            If the question is out of the domain of injection molding, simply answer that you do not have any information regarding this domain.
            [/INST] </s> 
            [INST] 
            Question: {question} 
            Context: {context} 
            Answer: 
            [/INST]
            """
        )
    
    def _prepare_chain_without_pdf(self):
        return ({"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_without_pdf
            | self.model
            | StrOutputParser())

    def ingest_pdf(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
            self._prepare_retriever()
        except Exception as e:
            logging.error(f"Error ingesting PDF: {e}")
            raise

    def _prepare_retriever(self):
        try:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.5},
            )
            self.chain_with_pdf = ({"context": self.retriever, "question": RunnablePassthrough()}
                                   | self.prompt_with_pdf
                                   | self.model
                                   | StrOutputParser())
        except Exception as e:
            logging.error(f"Error preparing retriever: {e}")
            raise

    def ask(self, query: str):
        try:
            if self.chain_with_pdf:
                return self.chain_with_pdf.invoke(query)
            else:
                return self.chain_without_pdf.invoke(query)
        except Exception as e:
            logging.error(f"Error processing user query: {e}")
            raise

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain_with_pdf = None
        self.chain_without_pdf = self._prepare_chain_without_pdf()


def setup_streamlit_page():
    st.set_page_config(page_title="Chat with Mistral chatbot", layout="wide")
    st.sidebar.title("Document Management")

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDFAssistant()


def display_chat_interface():
    st.subheader("Local Chatbot")
    chat_container = st.container()
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            message(msg, is_user=is_user, key=str(i))
    if "thinking_spinner" not in st.session_state:
        st.session_state["thinking_spinner"] = st.empty()


def handle_file_upload():
    if "assistant" in st.session_state:
        st.session_state["assistant"].clear()
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            file_path = temp_file.name

        if "ingestion_spinner" not in st.session_state:
            st.session_state["ingestion_spinner"] = st.empty()

        with st.session_state["ingestion_spinner"], st.spinner(f"Reading {file.name}"):
            st.session_state["assistant"].ingest_pdf(file_path)
        os.remove(file_path)


def setup_chat_page():
    setup_sidebar()
    st.header("Chat with Mistral chatbot")
    display_chat_interface()
    st.text_input("Message", key="user_input", on_change=process_user_input)


def setup_sidebar():
    st.sidebar.file_uploader("Upload new document", type=["pdf"], key="file_uploader",
                             on_change=handle_file_upload, label_visibility="collapsed",
                             accept_multiple_files=True)
    if st.sidebar.button("Clear Documents"):
        if "assistant" in st.session_state:
            st.session_state["assistant"].clear()
        st.session_state["messages"] = []


def process_user_input():
    user_text = st.session_state["user_input"].strip()
    if user_text:
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            try:
                agent_text = st.session_state["assistant"].ask(user_text)
            except Exception as e:
                agent_text = f"An error occurred: {e}"
                logging.error(f"Error processing user input: {e}")

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
    


if __name__ == "__main__":
    setup_streamlit_page()
    setup_chat_page()
