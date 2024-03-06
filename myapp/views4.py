# views.py
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from twilio.twiml.messaging_response import MessagingResponse
import requests

def get_result_from_query(query: str, zipName: str, prompt: str) -> str:
    url = 'https://nurenai2backend.azurewebsites.net/api/get-pdf/'
    headers = {'Content-Type': 'application/json'}

    data = {
        'message': query,
        'zipName': zipName,
        'prompt': prompt,
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response_data = response.json()
        return response_data.get('answer', '')
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return ''

@csrf_exempt
def incoming_sms(request):
    if request.method == 'POST':
        # Get the message the user sent to our Twilio number
        body = request.POST.get('Body', None)

        print(f"Received message: {body}")  # Add this print statement to check the received message

        # Start our TwiML response
        resp = MessagingResponse()

        # Initialize zipName and prompt if they are not already in the session
        if 'zipName' not in request.session:
            request.session['zipName'] = '11055e45-85e8-49c7-ab5c-f160d1733d88.zip'
        if 'prompt' not in request.session:
            request.session['prompt'] = 'Reply in 10 words to your disciple'

        # Determine the right reply for this message
        if body == 'Hello' or body == "Switch":
            # Send a message to the user to pick a mentor
            resp.message("Greetings! Please select a mentor to engage with:\n1. Krishna\n2. Steve Jobs\n3. Nietzsche\n4. Newton\n5. Napoleon\nType the corresponding number to select your mentor.")
            # Set the session state to indicate that we are waiting for the user's choice
            request.session['waiting_for_choice'] = True
        elif request.session.get('waiting_for_choice'):
            # User is expected to make a choice
            if body in ['1', '2', '3', '4', '5']:
                # User made a valid choice, update zipName and prompt based on the choice
                mentor_mapping = {
                    '1': ('11055e45-85e8-49c7-ab5c-f160d1733d88.zip', 'Reply in 10 words to your disciple'),
                    '2': ('1c16de68-d364-4dd2-bd47-7ad58bce3a60.zip', 'Reply in 10 words to an entrepreneur'),
                    '3': ('03101274-9092-472a-8c0c-89295c0c2c0c.zip', 'You are Zarathustra.Reply in 10 words to your student'),
                    '4': ('5af3a21b-6a1b-4caa-95fb-9a3387130960.zip', 'Reply in 10 words to a science student'),
                    '5': ('26623868-69c1-49cf-be37-4344eea7a688.zip', 'Reply in 10 words like you were a mentor')
                }
                zipName, prompt = mentor_mapping[body]
                request.session['zipName'] = zipName
                request.session['prompt'] = prompt
                result = get_result_from_query(body, zipName, prompt)
                resp.message(result)
            elif "Hello wizard of CS" in body:
                # Additional option for a different scenario
                request.session['zipName'] = 'archive.zip'
                request.session['prompt'] = 'Answer like you are Arpita'
                resp.message("Welcome, Wizard of CS! Your special mentor awaits you.")
            else:
                # User made an invalid choice
                resp.message('Invalid choice. Please pick a mentor by entering a number from 1 to 5.')
            # Reset the session state
            request.session.pop('waiting_for_choice', None)
        elif body == 'Bye':
            # User wants to end the conversation
            resp.message("Goodbye")
        else:
            # No specific action based on the user's input
            result = get_result_from_query(body, request.session['zipName'], request.session['prompt'])
            resp.message(result)

        print(f"Response sent: {str(resp)}")  # Add this print statement to check the response before returning

        # Return the TwiML response directly as an XML string
        return HttpResponse(str(resp), content_type='application/xml')
    else:
        print("Error: Only POST requests are allowed for this endpoint")
        return HttpResponse('<error>Only POST requests are allowed for this endpoint</error>', content_type='application/xml', status=405)
