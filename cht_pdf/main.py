import sys
import os
from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator


from dotenv import load_dotenv

load_dotenv()

query = sys.argv[1] 
print(query)

# llm = ChatOpenAI()

loader = TextLoader("data.txt")

# docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
response = index.query(query)
print(response)