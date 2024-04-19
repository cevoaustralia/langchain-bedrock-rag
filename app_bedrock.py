import os
from pathlib import Path
import boto3
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings

from langchain_astradb import AstraDBVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts.chat import ChatPromptTemplate
from langchain_astradb import AstraDBChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st
import tempfile

from helpers import website_to_txt

# External Dependencies:
import boto3
from botocore.config import Config

ASTRA_DB_APPLICATION_TOKEN = st.secrets["my_astradb_secrets"]["ASTRA_DB_APPLICATION_TOKEN"]

ASTRA_VECTOR_ENDPOINT = st.secrets["my_astradb_secrets"]["ASTRA_VECTOR_ENDPOINT"]
ASTRA_DB_ID = st.secrets["my_astradb_secrets"]["ASTRA_DB_ID"]
ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "cevostore"

#AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
#AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_secrets"]["AWS_DEFAULT_REGION"]
AWS_PROFILE = st.secrets["AWS_secrets"]["AWS_PROFILE"]

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

#os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
#os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
os.environ["AWS_PROFILE"] = AWS_PROFILE

os.environ["LANGCHAIN_PROJECT"] = "cevostore"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Started")


# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 4
top_k_memory = 3

###############
### Globals ###
###############

global lang_dict
global rails_dict
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory
global bedrock_runtime


#############
### Login ###
#############
# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.caption('Using a unique name will keep your content seperate from other users.')
            st.text_input('Username', key='username')
            #st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        #if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
        if len(st.session_state['username']) > 5:
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            #del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the username + password is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 Username must be 6 or more characters')
    return False

def logout():
    del st.session_state.password_correct
    del st.session_state.user
    del st.session_state.messages
    load_chat_history.clear()
    load_memory.clear()
    load_retriever.clear()
    

# Check for username/password and set the username accordingly
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

username = st.session_state.user


#######################
### Resources Cache ###
#######################

# Cache boto3 session for future runs
@st.cache_resource(show_spinner='Getting the Boto Session...')
def load_boto_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )
    return bedrock_client

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    #return OpenAIEmbeddings(model="text-embedding-3-small")
    # Get the Bedrock Embedding
    return BedrockEmbeddings(
        client=bedrock_runtime,
        #model_id="amazon.titan-embed-text-v1",
        model_id="cohere.embed-english-v3",
    )

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")
    # Get the load_vectorstore store from Astra DB
    return AstraDBVectorStore(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={
            "k": top_k_vectorstore,
            'filter': {'owner': username},
            }
    )

# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="anthropic.claude-v2"):
    print(f"load_model: {model_id}")
    # if model_id contains 'openai' then use OpenAI model
    if 'openai' in model_id:
        if '3.5' in model_id:
            gpt_version = 'gpt-3.5-turbo'
        else:
            gpt_version = 'gpt-4-turbo-preview'
        return ChatOpenAI(
            temperature=0.2,
            model=gpt_version,
            streaming=True,
            verbose=False
            )
    # else use Bedrock model
    return BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        streaming=True,
        #callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={"temperature": 0.2},
    )
# Anthropic Claude - NOT WORKING - required keys prompt, max_tokens_to_sample
# Amazon Titan - WORKING
# Meta Lllama - WORKING




# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history():
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )


# Focus on the user's needs and provide the best possible answer.
# You're friendly and you answer extensively with multiple sentences.
# You prefer to use bulletpoints to summarize.
# Do not include images in your response.
# Answer in English
# Cache prompt
@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You're a helpful AI assistant tasked to answer the user's questions
Do not include any information other than what is provied in the context below.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to provide a more personalized response:
{chat_history}

Question:
{question}
"""

    return ChatPromptTemplate.from_messages([("system", template)])


#################
### Functions ###
#################


# Function for Vectorizing uploaded data into Astra DB
def vectorize_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"Processing: {file.name}")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            print(f"Processing: {temp_filepath}")
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            # Create the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 100
            )

            if uploaded_file.name.endswith('txt'):
                file = [uploaded_file.read().decode()]
                texts = text_splitter.create_documents(file, [{'source': uploaded_file.name, 'owner': username}])
                vectorstore.add_documents(texts)
                st.info(f"{len(texts)} chunks loaded into Astra DB")            

            if uploaded_file.name.endswith('pdf'):
                # Read PDF
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)
                st.info(f"{len(pages)} pages loaded into Astra DB")

#
# Function for ingesting Text
#
def vectorize_text(text):
    print(f"Processing: {len(text)} characters")
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )
    # Split the text into chunks with Metadata
    texts = text_splitter.create_documents([text], [{'source': 'text', 'owner': username}])
    # Store the chunks in Astra DB
    vectorstore.add_documents(texts)
    st.info(f"{len(texts)} chunks loaded into Astra DB")

#
# Function for ingesting URLs
#
def vectorize_url(url):
    print(f"Processing: {url}")
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )
    # Get the text from the URL
    title, text = website_to_txt(url)
    if title == "Error":
        st.info(f"Error: {text}")
    else:
        # Split the text into chunks with Metadata
        texts = text_splitter.create_documents([text], [{'source': url, 'title': title, 'owner': username}])
        # Store the chunks in Astra DB
        vectorstore.add_documents(texts)
        st.info(f"{len(texts)} chunks loaded into Astra DB")

# Function for ingesting Youtube URLs
#
def vectorize_youtube_url(url):
    print(f"Processing: {url}")
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )
    # Get the text from the URL
    title, text = website_to_txt(url)
    if title == "Error":
        st.info(f"Error: {text}")
    else:
        # Split the text into chunks with Metadata
        texts = text_splitter.create_documents([text], [{'source': url, 'title': title, 'owner': username}])
        # Store the chunks in Astra DB
        vectorstore.add_documents(texts)
        st.info(f"{len(texts)} chunks loaded into Astra DB")

#
# Function for deleting an individual user's context
#
def delete_user_context(username):
    # Initialize the client
    from astrapy.db import AstraDB
    db = AstraDB(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )
    collection = db.collection(collection_name=ASTRA_DB_COLLECTION)
    # Delete the user's context
    deleted_count = collection.delete_many(filter={"metadata.owner": username})
    print(deleted_count)


#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="How may I help you today?")]



############
### Main ###
############


# Write the welcome text
st.markdown(Path('welcome.md').read_text())

# DataStax logo
with st.sidebar:
    st.image('./public/cevologo.webp')
    st.text('')

# Logout button
with st.sidebar:
    st.button(f"Logout '{username}'", on_click=logout)

# Initialize
with st.sidebar:
    bedrock_runtime = load_boto_client()
    embedding = load_embedding()
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    model = load_model()
    chat_history = load_chat_history()
    memory = load_memory()
    prompt = load_prompt()


# Sidebar
with st.sidebar:
    # Add Data Input options
    with st.container(border=True):
        st.caption('Load data using one of the methods below.')
        with st.form('load_files'):
            uploaded_files = st.file_uploader("Upload a file", type=["txt", "pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button('Upload')
            if submitted:
                with st.spinner('Chunking, Embedding, and Uploading to Astra'):
                    vectorize_files(uploaded_files)
        
        with st.form('load_website_url'):
            # option 2: enter URL
            url = st.text_input("Enter URL", "")
            submitted = st.form_submit_button('Embed')
            if submitted:
                if url is not None and url != "":
                    with st.spinner('Embedding and Uploading to Astra'):
                        vectorize_url(url)
        
        with st.form('load_youtube_url'):
            # option 2: enter URL
            url = st.text_input("Enter URL", "")
            submitted = st.form_submit_button('Embed')
            if submitted:
                if url is not None and url != "":
                    with st.spinner('Embedding and Uploading to Astra'):
                        vectorize_url(url)
        
        with st.form('load_text'):
            # option 3: enter text
            text = st.text_area("Enter Text", "")
            submitted = st.form_submit_button('Embed')
            if submitted:
                if text is not None and text != "":
                    with st.spinner('Embedding and Uploading to Astra'):
                        vectorize_text(text)


    # Add a drop down to choose the LLM model
    with st.container(border=True):
        model_id = st.selectbox('Choose the LLM model', [
            'meta.llama2-70b-chat-v1',
            'meta.llama2-13b-chat-v1',
            'amazon.titan-text-express-v1',
            'anthropic.claude-v2',
            #'anthropic.claude-3-sonnet-20240229-v1:0',
            #'openai.gpt-3.5',
            #'openai.gpt-4'
            ])
        model = load_model(model_id)


    # Drop the Chat History
    with st.form('delete_memory'):
        st.caption('Delete the conversational memory.')
        submitted = st.form_submit_button('Delete conversational memory')
        if submitted:
            with st.spinner('Delete chat history'):
                memory.clear()

    # Delete Context
    with st.form('delete_context'):
        st.caption("Delete the context")
        submitted = st.form_submit_button("Delete context")
        if submitted:
            with st.spinner("Removing context..."):
                delete_user_context(username)


# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input("What's up?"):
    print(f"Got question: \"{question}\"")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print("Display user prompt")
    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")

        # Customise for Anthropic models
        if 'anthropic' in model_id:
            system_message = "You are a helpful assistant."
            human_message = "Give me a joke"

            messages = [
                ("system", system_message),
                ("human", human_message)
            ]

        else:
            inputs = RunnableMap({
                'context': lambda x: retriever.get_relevant_documents(x['question']),
                'chat_history': lambda x: x['chat_history'],
                'question': lambda x: x['question']
            })
            chain = inputs | prompt | model
            print(f"Using inputs: {inputs}")

        
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        print(f"Response: {response}")
        content = response.content

        # Write the sources used
        relevant_documents = retriever.get_relevant_documents(question)
        if len(relevant_documents) > 0:
            content += f"""

*{"The following context was used for this answer:"}*  
"""
            sources = []
            for doc in relevant_documents:
                source = doc.metadata['source']
                page_content = doc.page_content
                #title = doc.metadata['title']
                if source not in sources:
                    content += f"""📙 :orange[{os.path.basename(os.path.normpath(source))}]  
"""
                    sources.append(source)
            print(f"Used sources: {sources}")

        # Write the final answer without the cursor
        response_placeholder.markdown(content)


        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))

# Add a space at bottom of screen
