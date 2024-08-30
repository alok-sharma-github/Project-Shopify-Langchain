from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS, cross_origin
from flask_cors import CORS
import uuid
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders.mongodb import MongodbLoader
from dotenv import load_dotenv
import nest_asyncio
import logging
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from openai import OpenAI
# openai = OpenAI(
#     api_key="~6eT/ycv50f_Mh5nFQ4QkbK3z6Xds4wz", # Refer to Create a secret key section
#     base_url="https://cloud.olakrutrim.com/v1",
# )

# def llm(message, session_id):
#     chat_completion = openai.chat.completions.create(
#         model="Meta-Llama-3-8B-Instruct",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": message}
#         ],
#         frequency_penalty=0,
#         logit_bias={2435: -100, 640: -100},
#         logprobs=True,
#         top_logprobs=2,
#         max_tokens=256,
#         n=1,
#         presence_penalty=0,
#         response_format={"type": "text"},
#         stop=None,
#         stream=False,
#         temperature=0,
#         top_p=1
#     )
#     return chat_completion['choices'][0]['message']['content']

# Flask App Initialization
app = Flask(__name__)
cors = CORS(app, resources={r"/query": {"origins": "http://localhost:5173"}})
app.secret_key = "b\xcb\x16}\xdd{~\x85\x7f\xae0\xb7\x17P\xd4Q\xa7\xf2v\xd4\xf0\xdb\xa3\xde\xdc"

# Load environment variables
nest_asyncio.apply()
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['MONGODB_URI'] = os.getenv('MONGODB_URI')  # Ensure this is set in your .env file

if not os.environ['GOOGLE_API_KEY']:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GoogleGenerativeAI and Embeddings
llm = OllamaFunctions(model="llama3.1", format="json")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
uri = os.getenv('MONGODB_URI')

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    logger.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error pinging MongoDB: {e}")

# Initialize document loader and create FAISS index
loader = MongodbLoader(
    connection_string=uri,
    db_name="shopifyData",
    collection_name="boatai",
    field_names=['_id', 'ProductId', 'Title', 'Description', 'Vendor', 'ProductType', 'Price', 'ImageURL', 'ProductURL'],
)

product_document = loader.load()
logger.info("Document Loaded")
db = FAISS.from_documents(product_document, embedding=embed_model)
logger.info("Vectors created")
db.save_local("faiss_index_books")
# db = FAISS.load_local("faiss_index_books", embedding=embedding_model)
retriever = db.as_retriever()

# Define prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. "
    "Ensure the reformulated question explicitly asks for details including URLs and Image URLs where relevant."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

prompt = (
    '''You are a conversational AI specializing in product recommendations.
    Whenever asked about recommendations include the product URL and image URL, 
    along with any other necessary details, and enhance the details accordingly from the context.
    Also always provide multiple options.
    Use only the provided context, **Do not use your own knowledge.**.
    Respond in HTML.

    ---------

    Example:
    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 400px;">
        <h3>{{ data.title }}</h3>
        <p><strong>Description:</strong> {{ data.description }}</p>
        <p><strong>Vendor:</strong> {{ data.vendor }}</p>
        <p><strong>Product Type:</strong> {{ data.product_type }}</p>
        <p><strong>Price:</strong> ₹{{ data.price }}</p>
        <a href="{{ data.product_url }}">
            <img src="{{ data.image_url }}" alt="{{ data.title }}" 
                 style="width: 75px; padding: 10px 5px; border-radius: 0%;"/>
        </a>
        <p><a href="{{ data.product_url }}" style="text-decoration:none;color:#0d6efd;">View Product</a></p>
    </div>

    ------------
    
    <context>
    {context}
    </context>
'''
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def send_message():
    if request.is_json:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400
 
        # Retrieve or set a session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
 
        print("Session ID: ", session_id)
 
        md_response_2 = conversational_rag_chain.invoke(
            {"input": message},
            config={
                "configurable": {"session_id": session_id}
            },
        )
 
        print("Output:\n", md_response_2)
        response_2 = {"reply": md_response_2['answer']}
        return jsonify(response_2)
    else:
        return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(debug=False)  # Change to False for production
