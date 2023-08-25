from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

origins = ["*"]



app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uvicorn.run(app, host="localhost", port=8100)