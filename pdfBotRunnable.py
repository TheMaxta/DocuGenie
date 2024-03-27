import os
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import os

def setup_environment(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = os.environ["OPENAI_API_KEY"]


def load_documents(directory):
    documents_reader = SimpleDirectoryReader(directory)
    documents = documents_reader.load_data()
    return documents

def create_index_from_documents(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index



def create_index_from_documents(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

def initialize_query_engine(index):
    return index.as_query_engine()


def execute_query(query_engine, query):
    return query_engine.query(query)

def main():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    directory = "PDF"
    query = "What can you tell me about vision transformers in the research paper? What contributions were made?"
    
    # Setup
    setup_environment(api_key)
    
    # Load Documents
    documents = load_documents(directory)
    
    # Create Index
    index = create_index_from_documents(documents)
    
    # Initialize Query Engine
    query_engine = initialize_query_engine(index)
    
    # Execute Query
    response = execute_query(query_engine, query)
    
    # Print Results
    print(response)

if __name__ == "__main__":
    main()
