import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from htmlTemplates import css, bot_template, user_template
import docx2txt
import streamlit_lottie as st_lottie
import json


def get_docs_text(docs):
    ''' 
    to extract text from a list of document files. Supports plain text (.txt), PDF (.pdf), and DOCX (.docx) formats.
    Takes a list of file objects and returns a concatenated string of their text content.
    ''' 
    text = ""
    for doc in docs:
        if doc is not None:
            if doc.type == "text/plain":  # txt doc
                text += str(doc.read(), encoding="utf-8")
            elif doc.type == "application/pdf":  # pdf
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                text += docx2txt.process(doc)
    return text


def get_text_chunks(text):
    '''
    to split a given text into smaller chunks.
    Uses CharacterTextSplitter to divide the text based on the specified separator, chunk size, and overlap.
    Takes a string as input and returns a list of text chunks.
 '''
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    ''' 
    to convert a list of text chunks into a vector store.
    Uses OpenAIEmbeddings to generate embeddings for each text chunk.
    The embeddings are stored in a FAISS index.
    Takes a list of text chunks as input and returns a FAISS vector store.
    '''
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    '''
    to create a conversational retrieval chain using a vector store.
    Initializes a ChatOpenAI model for generating responses.
    Uses ConversationBufferMemory to maintain the chat history.
    Constructs a ConversationalRetrievalChain that retrieves relevant information from the vector store.
    Takes a FAISS vector store as input and returns a conversational chain object.
    '''
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    '''
    to handle user input and manage the conversation flow.
    Checks if a conversation chain exists; if not, creates one using the vector store.
    Processes the user question and retrieves the response from the conversation chain.
    Displays the user's question and the bot's answer using custom templates.
    Updates the session state with the new chat history, ensuring no duplicate entries.
    Takes a user question as input.
    ''' 
    if st.session_state.conversation is None and st.session_state.vectorstore is not None:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
    
    response = st.session_state.conversation({"question": user_question})

    # Display current question and answer
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", response["answer"]), unsafe_allow_html=True)
    
    # Update session state with the new chat history, avoiding duplicates
    if not st.session_state.chat_history or (
        st.session_state.chat_history[-2].content != user_question and 
        st.session_state.chat_history[-1].content != response["answer"]
    ):
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))

def load_lottiefile(filepath: str):
    '''
    to load a Lottie animation file.
    Takes a file path as input and reads the JSON content of the Lottie file.
    Returns the JSON data for the Lottie animation. 
    '''
    with open(filepath, 'rb') as f:
        return json.load(f)

def main():
    '''
    Main function to run the Streamlit application. This function loads environment variables,
    sets the page configuration, and displays a Lottie animation. It initializes session state
    variables for conversation, chat history, and vector store. The function displays the main 
    header and input field for user questions, handling user input by creating a conversation 
    chain if necessary and processing the question. It also sets up a sidebar for document 
    upload, allowing users to upload multiple files, process them to extract text, create or 
    update the vector store, and display the chat history.
    '''
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books:")
    cover_pic = load_lottiefile('img/books.json')
    st.lottie(cover_pic, speed=0.5, reverse=False, loop=True, quality='low', height=200, key='first_animate')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple Documents 🔍")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None and st.session_state.vectorstore is not None:
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        if st.session_state.conversation:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv']
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_docs_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                # Create or update vector store
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                else:
                    new_vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore.merge(new_vectorstore)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        if st.button("Chat-History"):
            # Display's Chat History 
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.write(message.content)
                else:
                    st.write(message.content)

if __name__ == "__main__":
    main()
