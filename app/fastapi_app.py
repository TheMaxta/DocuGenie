from fastapi import FastAPI, Request
from main import fetch_response  # Assuming this is correctly implemented as async
from fastapi.middleware.cors import CORSMiddleware
import json  # Import the json module

app = FastAPI()

origins = [
    "http://localhost:7191",
    "https://localhost:7191",
    "https://localhost:7176",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/query/")
async def get_query(query: str):
    """GET endpoint to fetch a response for a given query passed as a URL parameter."""
    response = fetch_response(query)  # Make sure to await the async function call
    # Parse the JSON string to a Python dict if response is a string
    print(type(response))  # This will print the type of the response object
    cleaned_data = await clean_response(response)
    print(cleaned_data)
    return {"response": cleaned_data}

@app.post("/query/")
async def post_query(request: Request):
    """POST endpoint to fetch a response for a given query passed in the request body."""
    data = await request.json()
    query = data.get("InputText")
    response = fetch_response(query)  # Ensure to await the async function call
    # Parse the JSON string to a Python dict if response is a string
    print(type(response))  # This will print the type of the response object

    if isinstance(response, str):
        response = json.loads(response)
    cleaned_data = await clean_response(response)
    return {"response": cleaned_data}


""" SPECIAL NOTE: This method uses the llama-index <class 'llama_index.core.base.response.schema.Response'> object
    This means we return a string for response, and a string for formatted_sources.
    You may pass a truncation length to the method for longer source texts.
"""

async def clean_response(response):
    # Access the response text directly
    main_response = response.response  # Directly accessing the 'response' attribute
    formatted_sources = response.get_formatted_sources(length=250)  # Getting formatted sources
    
    # Return cleaned data
    return {
        'main_response': main_response,
        'formatted_sources': formatted_sources
    }
