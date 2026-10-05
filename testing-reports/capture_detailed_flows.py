import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
SAMPLE_IMAGE_PATH = os.path.abspath(r"..\backend\demo_data\breakhis_sample.png")

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self._msg_id = 0
        self.pending_responses = {}
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

    async def capture_screenshot(self, filename):
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        b64 = res.get("result", {}).get("data")
        if b64:
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f" -> Saved {filename} ({os.path.getsize(filepath):,} bytes)")
            return filepath
        return None

    async def upload_file(self, selector, filepath):
        doc = await self.send("DOM.getDocument")
        root_id = doc["result"]["root"]["nodeId"]
        node = await self.send("DOM.querySelector", {
            "nodeId": root_id,
            "selector": selector
        })
        node_id = node.get("result", {}).get("nodeId")
        if node_id:
            await self.send("DOM.setFileInputFiles", {
                "files": [filepath],
                "nodeId": node_id
            })
            return True
        return False

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            await self.ws.close()

async def run_detailed_flows():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    proc = subprocess.Popen([
        chrome_path,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--window-size=1440,1200",
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
        await client.send("DOM.enable")
        await client.send("Runtime.enable")

        print("=== DETAILED TEST FLOWS ===")

        # FLOW 1: Diagnostic Studio - Complete Image Upload & Quantum Encoding
        print("\n[Flow 1] Diagnostic Studio image upload & encoding...")
        await client.send("Page.navigate", {"url": "http://127.0.0.1:5173"})
        await asyncio.sleep(2.5)

        # Upload sample image
        await client.upload_file("input[type='file']", SAMPLE_IMAGE_PATH)
        await asyncio.sleep(1)

        # Click encode button
        await client.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const btn = btns.find(b => b.textContent.includes('Encode') && !b.disabled);
            if (btn) btn.click();
        """)
        print("Waiting 10s for quantum encoding to compute...")
        await asyncio.sleep(10)

        # Scroll to view results
        await client.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(1)
        await client.capture_screenshot("06_diagnostic_encode_success_full.png")

        # FLOW 2: Circuit Builder - Training and Curves
        print("\n[Flow 2] Circuit Builder tab and training...")
        await client.evaluate("""
            window.scrollTo(0, 0);
            document.querySelectorAll('nav button')[1].click();
        """)
        await asyncio.sleep(2)

        # Scroll to train model section
        await client.evaluate("""
            const trainSection = Array.from(document.querySelectorAll('h3')).find(h => h.textContent.includes('Train Model'));
            if (trainSection) trainSection.scrollIntoView();
        """)
        await asyncio.sleep(1)
        await client.capture_screenshot("07_circuit_builder_train_section.png")

        # Click Train Model
        print("Clicking Train Model button...")
        await client.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const trainBtn = btns.find(b => b.textContent.includes('Train Model') && !b.disabled);
            if (trainBtn) trainBtn.click();
        """)
        print("Waiting 8s for quantum model training...")
        await asyncio.sleep(8)
        await client.evaluate("""
            const trainSection = Array.from(document.querySelectorAll('h3')).find(h => h.textContent.includes('Train Model'));
            if (trainSection) trainSection.scrollIntoView();
        """)
        await asyncio.sleep(1)
        await client.capture_screenshot("09_circuit_builder_trained_curves.png")

        # FLOW 3: Noise Sandbox - Simulation Execution
        print("\n[Flow 3] Noise Sandbox simulation...")
        await client.evaluate("""
            window.scrollTo(0, 0);
            document.querySelectorAll('nav button')[2].click();
        """)
        await asyncio.sleep(2)

        # Enable both ZNE and TREX toggles if present
        print("Clicking Run Simulation...")
        await client.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const simBtn = btns.find(b => b.textContent.includes('Run Simulation') && !b.disabled);
            if (simBtn) simBtn.click();
        """)
        print("Waiting 6s for noise simulation...")
        await asyncio.sleep(6)
        await client.capture_screenshot("11_noise_simulation_results.png")

        # FLOW 4: Benchmark - Execution & Reports
        print("\n[Flow 4] Benchmark execution...")
        await client.evaluate("""
            window.scrollTo(0, 0);
            document.querySelectorAll('nav button')[3].click();
        """)
        await asyncio.sleep(2)

        print("Clicking Run Full Benchmark...")
        await client.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const runBtn = btns.find(b => b.textContent.includes('Run Full Benchmark') && !b.disabled);
            if (runBtn) runBtn.click();
        """)
        print("Waiting 8s for benchmark computation...")
        await asyncio.sleep(8)

        # Scroll to metrics
        await client.evaluate("window.scrollTo(0, 400);")
        await asyncio.sleep(1)
        await client.capture_screenshot("13_benchmark_results_charts.png")

        # Click Generate Report
        print("Clicking Generate Report...")
        await client.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const repBtn = btns.find(b => b.textContent.includes('Generate Report') && !b.disabled);
            if (repBtn) repBtn.click();
        """)
        await asyncio.sleep(4)
        await client.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(1)
        await client.capture_screenshot("14_benchmark_clinical_report_preview.png")

        await client.close()
    finally:
        proc.terminate()
        print("\n=== DETAILED TEST FLOWS COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(run_detailed_flows())
