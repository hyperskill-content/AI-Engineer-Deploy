import requests

url = "http://127.0.0.1:8000/ask"
payload = {
    "user_input": "Tell me about the nothing 1",
    "user_id": "test_user_1",
    "session_id": "session_abc123"
}

try:
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Response Text:", response.text)
except Exception as e:
    print(f"Request failed: {e}")