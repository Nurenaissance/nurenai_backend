
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_openai import ChatOpenAI
from django.http import JsonResponse
import json
import requests
from django.views.decorators.csrf import csrf_exempt
import os
import shutil
import logging
import faiss
from langchain_community.vectorstores import FAISS
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from langchain_community.callbacks import get_openai_callback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

prompt = hub.pull("rlm/rag-prompt")
import re
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=pdffornurenai;AccountKey=NfaInebhlvguuN9ZziAdwy1gyKZIfqmX1W1U1k/g/e0z1ZEsWqC7NXt8wSfWIQBusiN87/swIG95+AStJbrZTQ==;EndpointSuffix=core.windows.net")

container_name = "pdf"
container_client = blob_service_client.get_container_client(container_name)

logging.basicConfig(level=logging.INFO)
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
def query_pdf(query,prompt,zip_name_path):
    # Assuming that container_client is defined globally

    # Define paths
    # Define base paths
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
    #text_field = "document_type"
    embed = OpenAIEmbeddings(api_key="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", model="text-embedding-3-large")
    #index = pinecone.Index("sampledoc")
    loaded_vectorstore = FAISS.load_local(extracted_folder_path, embed,allow_dangerous_deserialization=True)
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
    # qa_chain = (
    # {
    #     "context": loaded_vectorstore.as_retriever(),
    #     "question": RunnablePassthrough(),
    # }
    # | prompt
    # | llm
    # | StrOutputParser()
    # )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(loaded_vectorstore.as_retriever(), combine_docs_chain)

    #rag_chain.invoke({"input": "What are autonomous agents?"})

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

    #custom_prompt=""
    custom_prompt=prompt
    
    #custom_prompt="Answer only in 15 words using the data from the document. Answer like you are krishna"
    # Combine the custom prompt with the user's query
    full_query = custom_prompt + query

    # Use asyncio to await the asynchronous call
    #result=vectorstore.similarity_search(
    # query,  # our search query
    # k=3  # return 3 most relevant docs
    #    )   
    with get_openai_callback() as cb:
        result = rag_chain.invoke({"input":full_query})
        print(cb.total_cost)
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
            prompt=data.get('prompt')
            #zip_name="e99204bd-5df1-49c7-b642-f667084b9633.zip"
            # Call your query_pdf function
            result =query_pdf(message,prompt,zip_name_path=zip_name)

            # You can modify this response format based on your needs
            final_answer = result.get('answer')

    # Prepare the response data
            response_data = {
                'answer': final_answer,
            }
            
            print(response_data)
            return JsonResponse(response_data)
       
@csrf_exempt
def phone_call(request):
        data = json.loads(request.body.decode('utf-8'))
        number=data.get('phone_number')
        zipname=data.get('zipName')
        backend_prompt=data.get('backend_prompt')
        url = "https://api.bland.ai/v1/calls"
        prompt = """
        BACKGROUND INFO: 
        You are an assistant. Be really kind to the client. If the user specifically ask for expert advise the response is{{backend_response}}
        """
        payload = {
            "phone_number": "+91" + number,
            "task": prompt,
            "model": "enhanced",
            "reduce_latency": True,
            "voice_id": 0,
            "tools":[{
                "name": "SendUserUtterance",
                "description": "Call for expert advise",
                "speech":"Hold on a second",
                "input_schema": {
                    "type": "object",
                    "properties": {
                    "transcript": {
                        "type":"string"
                    }
                    },
                    "required": ["transcript"]
                },
                "url": "https://127.0.0.1:8000/get-pdf/",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "message": "{{input.transcript}}",
                    "zipName":zipname,
                    "prompt":backend_prompt
                },        
                "response_data": [
                    {
                    "name": "backend_response",
                    "data": "$.answer"
                    }
                    ]
                }]
            }
        headers = {
            "authorization": "sk-7b7ga99r8bjlzd32o0gxm0cm4euirmjah50mzbxmt6rjcg0z05mm4jhmk29ckjfm69",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code == 200:
        # Return a JsonResponse with the appropriate content
            return JsonResponse({'message': 'success'})
        else:
        # Return an appropriate error response
            return JsonResponse({'error': 'An error occurred during the phone call request'}, status=500)