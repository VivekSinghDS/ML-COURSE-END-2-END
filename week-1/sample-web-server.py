from fastapi import FastAPI

app = FastAPI()

@app.get("/")
@app.get("/health_check")
async def health_check():
    return "ok"