# trial_sample.py

import socket

# 1. Resource leak
def fetch_api(endpoint):
    sock = socket.socket()
    sock.connect(('api.com', 80))
    sock.send(endpoint.encode())
    # Missing sock.close()
    return sock.recv(1024)

# 2. Null pointer dereference (None usage)
def process_data(data):
    if data is None:
        return data.upper()  # Will throw AttributeError

# 3. Concurrency issue
import threading

shared_counter = 0

def increment():
    global shared_counter
    for _ in range(1000):
        shared_counter += 1  # Not thread-safe

t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
# Missing t1.start() and t2.start() join for proper synchronization

# 4. Security vulnerability
import os

def read_secret_file(filename):
    return os.system("cat " + filename)  # Vulnerable to injection

# 5. Code complexity
def nested_loops():
    for i in range(5):
        for j in range(5):
            for k in range(5):
                print(i, j, k)