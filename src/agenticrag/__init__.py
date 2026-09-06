def main() -> None:
    """Start the local API service."""
    import uvicorn

    uvicorn.run("agenticrag.api.main:app", host="127.0.0.1", port=8000)
