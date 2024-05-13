from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
import os
import shutil

import pdfplumber
import PyPDF2
pdf_file = "English_NCERT_12.pdf"
def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text
text_content = pdf_to_text(pdf_file)

os.environ["OPENAI_API_KEY"] = "API_KEY_OF_YOURE_CHOICE"

#to split the text to create chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,  # Maximum characters per chunk
    chunk_overlap=50,  # Overlap between consecutive chunks (characters)
    length_function=len,  # Function to calculate chunk length (default: len)
    add_start_index=True  # Add starting index to each chunk (optional)
)

# Create documents (chunks) from the text file
chunks =text_splitter.split_text(text_content)

#To create embeddings
def create_embedding(text):
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",)
    embeddings = embed_model.embed_documents(text)
    return embeddings

documents_list = []
embeddings_list = []
ids_list = []
        
for i,chunk in enumerate(chunks):
    vector = create_embedding(chunk)

    documents_list.append(chunk)
    embeddings_list.append(vector)
    ids_list.append(f"English_NCERT_12_{i}") 

import chromadb
#to create a client instance
clients = chromadb.PersistentClient(path="This Pc/Downloads")    

# #creating a collection
collection = clients.get_or_create_collection("ENG_Collection")

#to add the data in the collections
collection.add(
    embeddings = embeddings_list,
    documents = documents_list,
    ids = ids_list
)

#Querying the collection
clients = chromadb.PersistentClient(path="This Pc/Downloads")
collection = clients.get_collection(name="ENG_Collection")

import chromadb
from langchain.llms import OpenAI

api_key = "AP_KEY_OF_YOURE_CHOICE"
llm = OpenAI(openai_api_key = api_key)

def process_query(query):
    #generate a prompt 
    prompt = f"Summarize this text {query}"
    response = llm.invoke(prompt)
    query_embeddings = create_embedding(response)

    # Retreive the infromation from chromadb
    collection = clients.get_collection("ENG_Collection")
    similar_documents = collection.query(query_embeddings)

    #processing from similar documents (chunks that is seperated to documents_list)
    processed_results = []
    for doc in similar_documents:
        prompt = f"Summarize this document: {doc['documents']}"
        answer = llm.invoke(prompt)
        processed_results.append({"id" : doc["id"] , "summary" : answer})
    
    return processed_results if processed_results else similar_documents["documents"]

#Example use case
user_query = "Why did the boy lost his ball?"
results = process_query(user_query)
print(results)