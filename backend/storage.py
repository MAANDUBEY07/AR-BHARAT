import os
from pathlib import Path

BASE = Path(__file__).parent
STORAGE = BASE / "storage"
STORAGE.mkdir(exist_ok=True)

def save_bytes(filename: str, b: bytes):
    """Save bytes to storage directory and return file path"""
    p = STORAGE / filename
    p.write_bytes(b)
    return str(p)

def list_storage():
    """List all files in storage directory"""
    return [str(p) for p in STORAGE.iterdir()]

def read_file(path):
    """Read file as bytes"""
    with open(path, "rb") as f:
        return f.read()