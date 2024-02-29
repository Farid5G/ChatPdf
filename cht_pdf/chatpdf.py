import sys
import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing_extensions import Concatenate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

pdf_reader = PdfReader("Langchain.pdf")

raw_text = ''
for i ,page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# print(raw_text)
        
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len
)

texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts,embeddings)

chain = load_qa_chain(OpenAI(),chain_type='stuff')

query = sys.argv[1]
docs = document_search.similarity_search(query)
response = chain.run(input_documents = docs,question= query)
print(response)