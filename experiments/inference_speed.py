import time
import json
import requests
from intent_detection import settings

if __name__ == "__main__":
    n = 100
    t = 0
    query = "Transfer 10 dollars to my savings account"
    for i in range(n):
        start_time = time.time()
        response = requests.get(f'{settings.IP_EC2}:8000/intent/intent?query="{query}"')
        t += time.time() - start_time
    print(f"time {(t / n):2.3f}s")
