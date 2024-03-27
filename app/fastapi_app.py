from fastapi import FastAPI
from main import fetch_response

app = FastAPI()


## EXAMPLE OF USE:

#http://localhost:8000/query/?query=YourQueryHere
#http://localhost:8000/query/?query=what is the difference between Cigna and Aetna?

@app.get("/query/")
async def query_endpoint(query: str):
    """Endpoint to fetch a response for a given query."""
    response = fetch_response(query)
    return {"response": response}
