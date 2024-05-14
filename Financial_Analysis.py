from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader

# loader = TextLoader("Enter the text file")
# data = loader.load()
# it will be in a list

loader = UnstructuredURLLoader(
    urls=["https://economictimes.indiatimes.com/"]
)

data = loader.load()
len(data) ## cuz only one link there
data[0].metadata

# Now creating the chunks of data , can using character spliting, recursive spllitting etc..., prefer to use recursive to avoid 
# sizing of chunk size due to seperators

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size = 200,
    chunk_overlap = 0
)

chunks = splitter.split_documents(data)
len(chunks)

# Now create a vector database, by converting into embeddings and storing them
import chromadb
import os

ids_list = []
documents_list = []

for i, chunk in enumerate(chunks):
    documents_list.append(chunk)
    ids_list.append(i)

#To create Embeddings
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "Your API key"

client = OpenAI()
embeddings_list = []

for i in range(0,len(embeddings_list), 2000):
    to_do_list = documents_list[i:2000+i]
    response = client.embeddings.create(
        input=to_do_list,
        model="text-embedding-ada-002"
    )
    new = []
    for i in response.data:
        new.append(i.embedding)
    embeddings_list += new

#Adding it to the vector base
client = chromadb.PersistentClient(".\chromadb")
#client.delete_collection(name="URL_Collections")
collection = client.get_or_create_collection(name="URL_Collections")

batch_size = 160  # Number of items in each batch

# Assuming len(embeddings_list) == len(documents_list) == len(metadata_list) == len(ids_list)
total_items = len(embeddings_list)

for start_idx in range(0, total_items, batch_size):
    end_idx = min(start_idx + batch_size, total_items)
    
    batch_embeddings = embeddings_list[start_idx:end_idx]
    batch_documents = documents_list[start_idx:end_idx]
    batch_ids = ids_list[start_idx:end_idx]
    
    # Add the batch to the collection
    collection.add(embeddings=batch_embeddings,
                   documents=batch_documents,
                   ids=batch_ids)
    

'''
batch size is set bacause a chromadb collection can only take 166 at a time
This is only to upload to chromadb, for the retrieval and generation part,
you have to always build a custom one to suit your requirement
metadata_list can also be custom depending on how you want

You can also do query processing using collections.query an
'''