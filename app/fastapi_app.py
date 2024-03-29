from fastapi import FastAPI, Request
from main import fetch_response
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
 
origins = [
    "http://localhost:7191",
    "https://localhost:7191",
    "https://localhost:7176"
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



## EXAMPLE OF USE:

#http://localhost:8000/query/?query=YourQueryHere
    #http://localhost:8000/query/?query=what&is&thedifference&between&Cigna&and&Aetna?


@app.get("/query/")
async def get_query(query: str):
    """GET endpoint to fetch a response for a given query passed as a URL parameter."""
    response = fetch_response(query)
    return {"response": response}

@app.post("/query/")
async def post_query(request: Request):
    """POST endpoint to fetch a response for a given query passed in the request body."""
    data = await request.json()
    query = data.get("InputText")
    response = fetch_response(query)
    return {"response": response}