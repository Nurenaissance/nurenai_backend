from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import ImageCaptionLoader
import os
import shutil
import uuid
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import Docx2txtLoader

blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=pdffornurenai;AccountKey=NfaInebhlvguuN9ZziAdwy1gyKZIfqmX1W1U1k/g/e0z1ZEsWqC7NXt8wSfWIQBusiN87/swIG95+AStJbrZTQ==;EndpointSuffix=core.windows.net")
container_name = "pdf"
container_client = blob_service_client.get_container_client(container_name)
def download_blob(blob_client, local_file_name):
    with open(local_file_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

@csrf_exempt
@require_POST
def upload_pdf_view(request):
    uploaded_file = request.FILES.get('file')
    unique_filename = str(uuid.uuid4()) + os.path.splitext(uploaded_file.name)[1]

    blob_client = container_client.get_blob_client(unique_filename)
    blob_client.upload_blob(uploaded_file)
    local_file_name = os.path.join(r"C:\Users\Adarsh\MyProject\lang\vectorstores", unique_filename)
    download_blob(blob_client, local_file_name)
    file_path = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{unique_filename}"
    if uploaded_file.name.lower().endswith('.pdf'):
        loader = PyPDFLoader(local_file_name)
    elif uploaded_file.name.lower().endswith('.docx'):
        loader = Docx2txtLoader(local_file_name)
    elif uploaded_file.name.lower().endswith('.csv'):
        loader = CSVLoader(local_file_name)
    elif uploaded_file.name.lower().endswith(('.jpg', '.png', '.bmp')):
        loader = UnstructuredImageLoader(local_file_name)
    else:
        return JsonResponse({'error': 'Unsupported file format'})

    documents = loader.load()
    os.remove(local_file_name)
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    local_folder_path = os.path.join("C:\\Users\\Adarsh\\MyProject\\lang\\vectorstores", str(uuid.uuid4()))
    vectorstore.save_local(local_folder_path)

    zip_file_path = shutil.make_archive(local_folder_path, 'zip', local_folder_path)
    zip_blob_name = os.path.basename(zip_file_path)
    
    with open(zip_file_path, "rb") as data:
        blob_client = container_client.get_blob_client(zip_blob_name)
        blob_client.upload_blob(data)

    return JsonResponse({'message': 'File uploaded successfully', 'filename': unique_filename, 'zip_file_path': zip_blob_name})
