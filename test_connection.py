import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key or not index_name:
    raise ValueError("Check your .env file.")

pc = Pinecone(api_key=api_key)

index = pc.Index(index_name)

print("Pinecone connected successfully!")

print(index.describe_index_stats())