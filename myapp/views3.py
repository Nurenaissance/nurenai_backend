from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI

# Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "product": {"type": "string"},
        "price": {"type": "integer"},
    },
    "required": ["name", "product"],
}




from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import os
import shutil
import traceback
import logging
import uuid
from langchain.vectorstores import FAISS
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.chains import RetrievalQAWithSourcesChain
from openai import OpenAI
client = OpenAI()
OpenAI.api_key = "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"
from django.utils import timezone
import uuid 
import traceback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


#from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA


blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=pdffornurenai;AccountKey=NfaInebhlvguuN9ZziAdwy1gyKZIfqmX1W1U1k/g/e0z1ZEsWqC7NXt8wSfWIQBusiN87/swIG95+AStJbrZTQ==;EndpointSuffix=core.windows.net")

container_name = "pdf"
container_client = blob_service_client.get_container_client(container_name)

logging.basicConfig(level=logging.INFO)

@csrf_exempt
def download_and_extract_zip(blob_client, downloaded_zip_path, extracted_folder_path):
    try:
        with open(downloaded_zip_path, "wb") as data:
            data.write(blob_client.download_blob().readall())

        # Add logging to identify when the extraction starts
        logging.info(f"Starting extraction from {downloaded_zip_path} to {extracted_folder_path}")

        shutil.unpack_archive(downloaded_zip_path, extracted_folder_path)

        # Add logging to identify when the extraction is successful
        logging.info(f"Extraction successful")
    except Exception as e:
        # Add logging to capture the exception details
        logging.error(f"Error during download and extraction: {e}")

    
@csrf_exempt
def query_pdf(query,zip_name):
    # Assuming that container_client is defined globally
    zip_name_path=zip_name
    # Define paths
    base_path = r"C:\Users\Adarsh\MyProject\lang\vectorstores"
    
    # Extract zip name without extension
    zip_name = os.path.splitext(os.path.basename(zip_name_path))[0]

    # Define paths based on zip name
    downloaded_zip_path = os.path.join(base_path, f"downloaded_{zip_name}.zip")
    extracted_folder_path = os.path.join(base_path, f"downloaded_{zip_name}")
    #subfolder_name = os.path.splitext(os.path.basename(zip_name_path))[0]
    #subfolder_path = os.path.join(extracted_folder_path, subfolder_name)
    # Check if the data has already been downloaded and extracted
    if not os.path.exists(downloaded_zip_path) or not os.path.exists(extracted_folder_path):
        # Download and extract the zip file
        blob_name = f"{zip_name_path}"
        blob_client = container_client.get_blob_client(blob_name)
        download_and_extract_zip(blob_client, downloaded_zip_path, extracted_folder_path)

    # Continue with the rest of your query logic
   
    embed = OpenAIEmbeddings("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")
    
    loaded_vectorstore = FAISS.load_local(extracted_folder_path, embed)
    #vectorstore = Pinecone(
     # index, embed , text_field
    #)
    #   metadata_condition = {"document_type": "gita"}  # Specify the metadata condition you want to match
    #custom_retriever = CustomMetadataRetriever(vectorstore=vectorstore, metadata_condition=metadata_condition)

    llm = ChatOpenAI(
    openai_api_key="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",
    model_name='gpt-3.5-turbo',
    temperature=0.0
    )
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    #retriever=custom_retriever
    #retriever=vectorstore.as_retriever()
    retriever=loaded_vectorstore.as_retriever()
    )
    #qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    #llm=llm,
    #chain_type="stuff",
    #retriever=loaded_vectorstore.as_retriever(search_type="mmr")
    #)
    

    #embeddings = OpenAIEmbeddings()

    # Load from local storage

    #persisted_vectorstore = FAISS.load_local(r"C:\Users\Adarsh\MyProject\lang\backend\gita_index_constitution", embeddings)

    # Use RetrievalQA chain for orchestration
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())

    custom_prompt=""
    #custom_prompt="Answer only in 15 words using the data from the document. Answer like you are krishna"
    # Combine the custom prompt with the user's query
    full_query = custom_prompt + query

    # Use asyncio to await the asynchronous call
    #result=vectorstore.similarity_search(
    # query,  # our search query
    # k=3  # return 3 most relevant docs
    #    )   
    result = qa.run(full_query)
    #result=qa_with_sources(full_query)

    # Return the result
    return result

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST' and 'audio' in request.FILES:
        audio_file = request.FILES['audio']
        zip_name=request.POST.get('zipName')
        # Define the directory where you want to save the audio files
        save_directory = r'C:\Users\Adarsh\MyProject\lang\data\audio'
        print("sdvsv: ",zip_name)
        # Ensure the directory exists, create it if not
        os.makedirs(save_directory, exist_ok=True)

        # Generate a unique filename with a .wav extension
        custom_filename = f'{uuid.uuid4().hex}.wav'
        
        # Save the audio file locally with the custom filename
        save_path = os.path.join(save_directory, custom_filename)
        with open(save_path, 'wb') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        try:
            # Transcribe the audio using OpenAI's Whisper API
            response = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=open(save_path, 'rb')
                            )

            # Get the transcription from the API response
            transcription = response.text
            print(transcription)
            # Call the query_pdf function with the transcription
            result_from_query = query_pdf(transcription,zip_name)
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            chain = create_extraction_chain(schema, llm)
            result=chain.run(transcription)
          

            # Include the result from query_pdf in the JSON response
            return JsonResponse({'status': 'success', 'transcription': transcription, 'result_from_query': result_from_query, 'schema':result})  
        except Exception as e:
            # Print the traceback information for debugging
            traceback.print_exc()

            return JsonResponse({'error': f'Error processing audio: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)
