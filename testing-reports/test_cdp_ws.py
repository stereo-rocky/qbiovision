import asyncio
import base64
import json
import subprocess
import time
import urllib.request
import websockets

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self._msg_id = 0
        self.pending_responses = {}
        self.events = []
        self._listener_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50_000_000)
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    future = self.pending_responses.pop(msg["id"], None)
                    if future and not future.done():
                        future.set_result(msg)
                else:
                    self.events.append(msg)
        except Exception:
            pass

    async def send(self, method, params=None):
        self._msg_id += 1
        msg_id = self._msg_id
        future = asyncio.get_event_loop().create_future()
        self.pending_responses[msg_id] = future
        payload = {"id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params
        await self.ws.send(json.dumps(payload))
        return await future

    async def evaluate(self, expression):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True
        })
        return res.get("result", {}).get("result", {}).get("value")

    async def capture_screenshot(self, filepath):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        b64 = res.get("result", {}).get("data")
        if b64:
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(b64))
            return True
        return False

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            await self.ws.close()

async def test():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    proc = subprocess.Popen([
        chrome_path,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--window-size=1400,900",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank"
    ])
    try:
        await asyncio.sleep(2)
        with urllib.request.urlopen("http://127.0.0.1:9222/json") as response:
            targets = json.loads(response.read().decode())
        ws_url = None
        for t in targets:
            if t.get("type") == "page":
                ws_url = t.get("webSocketDebuggerUrl")
                break
        
        client = CDPClient(ws_url)
        await client.connect()
        await client.send("Page.enable")
        await client.send("Runtime.enable")
        print("Navigating to local site...")
        await client.send("Page.navigate", {"url": "http://127.0.0.1:5173"})
        await asyncio.sleep(3)
        
        title = await client.evaluate("document.title")
        print(f"Document Title: {title}")
        
        ok = await client.capture_screenshot("screenshots/cdp_test_success.png")
        print(f"Screenshot captured: {ok}")
        await client.close()
    finally:
        proc.terminate()

asyncio.run(test())
