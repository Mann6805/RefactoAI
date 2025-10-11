# trial_mixed.py

import socket
import threading

# --------- Defective functions ---------
# Resource leak
def fetch_api(endpoint):
    sock = socket.socket()
    sock.connect(('api.com', 80))
    sock.send(endpoint.encode())
    # Missing sock.close()
    return sock.recv(1024)

# Null pointer dereference
def process_data(data):
    if data is None:
        return data.upper()  # Will throw AttributeError

# Concurrency issue
shared_counter = 0
def increment():
    global shared_counter
    for _ in range(1000):
        shared_counter += 1  # Not thread-safe

# --------- Correct functions ---------
def safe_fetch_api(endpoint):
    sock = socket.socket()
    try:
        sock.connect(('api.com', 80))
        sock.send(endpoint.encode())
        return sock.recv(1024)
    finally:
        sock.close()  # Properly closed

def process_text(data):
    if data is None:
        return ""
    return data.upper()  # Safe

def increment_safe(lock):
    global shared_counter
    for _ in range(1000):
        with lock:
            shared_counter += 1  # Thread-safe
