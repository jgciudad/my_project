from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the MNIST model inference API!"}

