# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
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


@csrf_exempt
def query_pdf(query):
    embeddings = OpenAIEmbeddings(openai_api_key="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")

    # Load from local storage
    persisted_vectorstore = FAISS.load_local(r"C:\Users\Adarsh\MyProject\lang\backend\gita_index_constitution", embeddings)

    # Use RetrievalQA chain for orchestration
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())

    custom_prompt="Answer as if you are krishna guiding arjuna. Answer in 15 words"
    # Combine the custom prompt with the user's query
    full_query = custom_prompt + query

    # Use asyncio to await the asynchronous call
    result = qa.run(full_query)

    # Return the result
    return result

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST' and 'audio' in request.FILES:
        audio_file = request.FILES['audio']

        # Define the directory where you want to save the audio files
        save_directory = r'C:\Users\Adarsh\MyProject\lang\data\audio'

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

            # Call the query_pdf function with the transcription
            result_from_query = query_pdf(transcription)

            # Include the result from query_pdf in the JSON response
            return JsonResponse({'status': 'success', 'transcription': transcription, 'result_from_query': result_from_query})  
        except Exception as e:
            # Print the traceback information for debugging
            traceback.print_exc()

            return JsonResponse({'error': f'Error processing audio: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)
