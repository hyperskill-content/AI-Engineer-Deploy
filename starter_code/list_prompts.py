from langfuse import get_client
import dotenv
dotenv.load_dotenv()

client = get_client()
prompts = client.api.prompts.list()
for p in prompts.data:
    print(f"Name: {p.name}")
