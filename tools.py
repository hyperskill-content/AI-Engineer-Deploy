import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import dotenv
import os
dotenv.load_dotenv()

class Tools:
    def __init__(self):
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            show_progress_bar=True,
        )

    def embed_documents(self, json_path: str):
        """
        Load JSON data from the specified file and convert each entry to a Document.
        :param
            json_path (str): Path to the JSON file containing smartphone data.

        :returns
            Chroma: A Chroma vector store built from the smartphone documents,
                    or an empty list if an error occurs.
        """

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {json_path} was not found.")
            return []
        except json.JSONDecodeError as jde:
            print(f"Error decoding JSON from file {json_path}: {jde}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while reading {json_path}: {e}")
            return []

        documents = []
        for entry in data:
            # Build a readable content string from the JSON entry
            content = (
                f"Model: {entry.get('model', '')}\n"
                f"Price: {entry.get('price', '')}\n"
                f"Rating: {entry.get('rating', '')}\n"
                f"SIM: {entry.get('sim', '')}\n"
                f"Processor: {entry.get('processor', '')}\n"
                f"RAM: {entry.get('ram', '')}\n"
                f"Battery: {entry.get('battery', '')}\n"
                f"Display: {entry.get('display', '')}\n"
                f"Camera: {entry.get('camera', '')}\n"
                f"Card: {entry.get('card', '')}\n"
                f"OS: {entry.get('os', '')}\n"
                f"In Stock: {entry.get('in_stock', '')}"
            )
            documents.append(Document(page_content=content))

        try:
            collection_name = "smartphones"


            collection_exists = self.qdrant_client.collection_exists(collection_name=collection_name)
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                    ),
                )

                qdrant_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embedding=self.embeddings_model
                )

                qdrant_store.add_documents(documents=documents)

                return qdrant_store

            # no need to create a vector store every time
            else:
                qdrant_store = QdrantVectorStore.from_existing_collection(
                    embedding=self.embeddings_model,
                    collection_name=collection_name,
                )

                return qdrant_store

        except Exception as e:
            print(f"Error initializing the vector store: {e}")
            return []

    # ---------------------------
    # Tool Definitions
    # ---------------------------
    #@tool("SmartphoneInfo")
    def smartphone_info_tool(self, model: str) -> str:
        """
         Process tool calls from the language model and collect their responses.

         :param
             llm_with_tools: The language model instance with bound tools.

         :returns
             Toolresponse
         """
        try:
            if self.qdrant_client.collection_exists("smartphones"):
                # 2. Get the collection information
                # collection_info = qdrant_client.(collection_name)
                qdrant_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name="smartphones",
                    embedding=self.embeddings_model
                )
                results = qdrant_store.similarity_search(model, k=1)
                if not results:
                    print(f"Info: No results found for model: {model}")
                    return "Could not find information for the specified model."
                info = results[0].page_content
                return info
            else:
                print(f"Collection does not exist.")
        except Exception as e:
            print(f"Error during smartphone information retrieval for model {model}: {e}")
            return f"Error during smartphone information retrieval: {e}"

