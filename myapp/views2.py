
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import os
import shutil
import faiss
from langchain.vectorstores import FAISS
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.chains import RetrievalQAWithSourcesChain


#from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from openai import OpenAI
import pinecone
import re
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=pdffornurenai;AccountKey=NfaInebhlvguuN9ZziAdwy1gyKZIfqmX1W1U1k/g/e0z1ZEsWqC7NXt8wSfWIQBusiN87/swIG95+AStJbrZTQ==;EndpointSuffix=core.windows.net")
pinecone.init(api_key="d7af7a08-e691-4789-810d-4e1274fd7080", environment="gcp-starter")
container_name = "pdf"
container_client = blob_service_client.get_container_client(container_name)

'''class CustomMetadataRetriever:
    def __init__(self, vectorstore: Pinecone, metadata_condition: dict):
        self.vectorstore = vectorstore
        self.metadata_condition = metadata_condition

    def get_relevant_documents(self, question: str) -> list[tuple[str, float]]:
        # Retrieve all documents from the vectorstore
        all_documents = self.vectorstore.similarity_search(query=question)

        # Filter documents based on metadata condition
        filtered_documents = [(id, score) for id, score in all_documents if self._metadata_matches(id)]

        return filtered_documents

    def _metadata_matches(self, document_id: str) -> bool:
        # Retrieve metadata for the document from the vectorstore
        document_metadata = self.vectorstore.get_metadata(document_id)

        # Check if metadata matches the specified condition
        return all(document_metadata.get(key) == value for key, value in self.metadata_condition.items())
'''
@csrf_exempt
def download_and_extract_zip(blob_client, downloaded_zip_path, extracted_folder_path):
    with open(downloaded_zip_path, "wb") as data:
        data.write(blob_client.download_blob().readall())

    shutil.unpack_archive(downloaded_zip_path, extracted_folder_path)
@csrf_exempt
def query_pdf(query, zip_name_path):
    # Assuming that container_client is defined globally

    # Define paths
    downloaded_zip_path = r"C:\Users\Adarsh\MyProject\lang\vectorstores\downloaded_gita_index_constitution.zip"
    extracted_folder_path = r"C:\Users\Adarsh\MyProject\lang\vectorstores\downloaded_gita_index_constitution"

    # Check if the data has already been downloaded and extracted
    if not os.path.exists(downloaded_zip_path) or not os.path.exists(extracted_folder_path):
        # Download and extract the zip file
        blob_name = f"C:/Users/Adarsh/MyProject/lang/vectorstores/{zip_name_path}"
        blob_client = container_client.get_blob_client(blob_name)
        download_and_extract_zip(blob_client, downloaded_zip_path, extracted_folder_path)

    # Continue with the rest of your query logic
    text_field = "document_type"
    embed = OpenAIEmbeddings()
    index = pinecone.Index("sampledoc")
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
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=loaded_vectorstore.as_retriever(search_type="mmr")
    )
    #embeddings = OpenAIEmbeddings()

    # Load from local storage

    #persisted_vectorstore = FAISS.load_local(r"C:\Users\Adarsh\MyProject\lang\backend\gita_index_constitution", embeddings)

    # Use RetrievalQA chain for orchestration
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())

    #custom_prompt="Answer as if you are krishna guiding arjuna. Answer in 10 words"
    custom_prompt="Answer only in 15 words using the data from the document. Answer like you are krishna"
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
def query_pdf_view(request):
            # Load JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))

            # Assuming the message is sent in the 'message' field
            message = data.get('message')
            zip_name=data.get('zipName')
            # Call your query_pdf function
            result =query_pdf(message,zip_name_path=zip_name)

            # You can modify this response format based on your needs
            response_data = {
                'answer': result,
            }

            return JsonResponse(response_data)
       
