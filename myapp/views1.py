

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from openai import OpenAI
import shutil
import os
import re
from langchain.vectorstores import FAISS
import openai
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
client = OpenAI()
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import uuid  # Import the uuid module for generating unique filenames
import pinecone
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=pdffornurenai;AccountKey=NfaInebhlvguuN9ZziAdwy1gyKZIfqmX1W1U1k/g/e0z1ZEsWqC7NXt8wSfWIQBusiN87/swIG95+AStJbrZTQ==;EndpointSuffix=core.windows.net")
pinecone.init(api_key="d7af7a08-e691-4789-810d-4e1274fd7080", environment="gcp-starter")
index = pinecone.Index("sampledoc")
container_name = "pdf"
container_client = blob_service_client.get_container_client(container_name)
model_name = 'text-embedding-ada-002'


@csrf_exempt
@require_POST
#def upload_pdf_view(request):
    #index = pinecone.Index("sampledoc")
    #container_name = str(uuid.uuid4())

# Create the container
    #container_client = blob_service_client.create_container(container_name)
    # Assuming the file is sent as 'pdf_file' in the form data
    #uploaded_file = request.FILES.get('pdf_file')

    # Specify the directory where you want to save the uploaded PDF
    #save_directory = 'C:\\Users\\Adarsh\\MyProject\\lang\\data'

    # Ensure the directory exists, create it if it doesn't
    #if not os.path.exists(save_directory):
    #    os.makedirs(save_directory)

    # Generate a unique filename using uuid
    #unique_filename = str(uuid.uuid4()) + '.pdf'

    # Construct the file path using the unique filename
    #file_path = os.path.join(save_directory, unique_filename)

    # Open the file and save the content
    #with open(file_path, 'wb') as pdf_file:
    #    for chunk in uploaded_file.chunks():
    #        pdf_file.write(chunk)
    
    # Your logic to process the uploaded PDF goes here
    #custom logic for pdf processing
    #loader = PyPDFLoader(file_path)
    #documents = loader.load()
    # Split document in chunks
   # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #docs = text_splitter.split_documents(documents)
    # Convert Document objects into strings
    
    #texts = [str(doc) for doc in docs]
    #ids=[file_path]
    #embeddings = OpenAIEmbeddings()
    #embeddings_list = []
    #for text in texts:
    #    res = client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
    #    embeddings_list.append(res) 
    #embeddings=embeddings_list
    #index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])
    
    # Create vectors
    #vectorstore = FAISS.from_documents(docs, embeddings)
    # Persist the vectors locally on disk
    #vectorstore.save_local(r"C:\Users\Adarsh\MyProject\lang\backend\gita_index_constitution")
    
    
    #index = pinecone.Index('sampledoc')
    #docsearch = pinecone.from_documents([t.page_content for t in docs], embeddings, index_name="sampledoc")
    #logic ends
    # For example, you can perform further processing or return additional information

    # Send a JSON response with the unique filename
    #return JsonResponse({'message': 'File uploaded successfully', 'filename': unique_filename})
def upload_pdf_view(request):
    # Assuming the file is sent as 'pdf_file' in the form data
    uploaded_file = request.FILES.get('pdf_file')

    # Generate a unique filename using uuid
    unique_filename = str(uuid.uuid4()) + '.pdf'
    unique_filename1= str(uuid.uuid4())

    # Upload the PDF to Azure Blob Storage
    blob_client = container_client.get_blob_client(unique_filename)
    blob_client.upload_blob(uploaded_file)

    # Your logic to process the uploaded PDF goes here
    # Custom logic for pdf processing
    file_path = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{unique_filename}"

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    embeddings=OpenAIEmbeddings()
    # Split document in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    local_folder_path = os.path.join(r"C:\Users\Adarsh\MyProject\backend_nurenai\backend\myproject\vectorstore", unique_filename1)
    vectorstore.save_local(local_folder_path)
    zip_file_path = local_folder_path
    shutil.make_archive(zip_file_path, 'zip', local_folder_path)
    
    blob_name = zip_file_path
    zip_file_path=zip_file_path+'.zip'
    blob_client = container_client.get_blob_client(blob_name)
    with open(zip_file_path, "rb") as data:
        blob_client.upload_blob(data)

    # Convert Document objects into strings
    #texts = [str(doc) for doc in docs]
    #ids = [file_path]
    #metadatas=[{"document_type": "gita"}]
    #embeddings_list = []
    #for text in texts:
        #res = client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
        #embeddings_list.append(res)

    #embeddings = embeddings_list
    #index.upsert(vectors=[(id, embedding, metadata) for id, embedding, metadata in zip(ids, embeddings, metadatas)])

    # For example, you can perform further processing or return additional information

    # Send a JSON response with the unique filename
    return JsonResponse({'message': 'File uploaded successfully', 'filename': unique_filename,'zip_file_path':unique_filename1})