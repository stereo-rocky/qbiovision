import urllib.request
import json
import subprocess
import time
import os

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
proc = subprocess.Popen([
    chrome_path,
    "--headless=new",
    "--remote-debugging-port=9222",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "http://127.0.0.1:5173"
])

print(f"Chrome launched with PID: {proc.pid}")
time.sleep(3)

try:
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as response:
        data = json.loads(response.read().decode())
        print("CDP targets found:")
        for target in data:
            print(" -", target.get("title"), target.get("url"), target.get("webSocketDebuggerUrl"))
finally:
    proc.terminate()
    print("Chrome terminated.")
