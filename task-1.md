## How to run?

```bash
pip install -r requirements
python uvicorn main:app --reload
```

Send an example request:
```bash
curl --request POST \
  --url http://127.0.0.1:8000/ask \
  --header 'content-type: application/json' \
  --data '{
  "user_input": "Compare OnePlus 11 5G and OPPO Reno 9 Pro Plus camera",
  "user_id": "Mike",
  "session_id": "mb-1"
}'
```

![Request-Response](task-1-response.png)