#!/usr/bin/env python3
"""Test the download endpoint directly"""

from app import app
import requests
import time

def test_endpoint():
    # Test the endpoint
    try:
        response = requests.get('http://localhost:5000/api/patterns/14/download?format=png', timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content length: {len(response.content)}")
        if response.status_code != 200:
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == '__main__':
    test_endpoint()