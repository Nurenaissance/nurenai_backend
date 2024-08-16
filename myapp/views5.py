from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json

@csrf_exempt
def incoming_call(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        phone_number = data.get('phone_number')
        zipName = data.get('zipName')
        backend_prompt = data.get('backend_prompt')
        url = "https://api.bland.ai/v1/calls"
        prompt = """
        BACKGROUND INFO: 
        You are krishna. Be really kind to the client. If the user specifically ask for krishna the response is{{backend_response}}
        Talk really kindly and softly and mention about the greatness of Gita.
        """
        payload = {
            "phone_number": phone_number,
            "task": prompt,
            "model": "enhanced",
            "reduce_latency": True,
            "voice_id": 0,
            "tools":[{
                "name": "SendUserUtterance",
                "description": "Call for expert advise from krishna",
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
                "url": "https://127.0.0.1:8000/api/get-pdf/",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "message": "{{input.transcript}}",
                    "zipName":zipName,
                    "prompt":""
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

        return JsonResponse(response.json())
    else:
        return JsonResponse({'error': 'Only POST requests are allowed for this endpoint'}, status=405)
