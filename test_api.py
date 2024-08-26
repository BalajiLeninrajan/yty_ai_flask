
import requests

response = requests.post(
    "http://localhost:5000/",
    json={
        "prompt" : "tell me a joke"
    }
)

print(response.json())
