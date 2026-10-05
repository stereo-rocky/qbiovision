import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets

async def test():
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
    await asyncio.sleep(2)
    try:
        with urllib.request.urlopen("http://127.0.0.1:9222/json") as r:
            targets = json.loads(r.read().decode())
        ws_url = [t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page"][0]
        
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            msg_id = 0
            async def send(method, params=None):
                nonlocal msg_id
                msg_id += 1
                curr_id = msg_id
                payload = {"id": curr_id, "method": method}
                if params: payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    if msg.get("id") == curr_id:
                        return msg

            await send("Page.enable")
            await send("Runtime.enable")
            await send("Page.navigate", {"url": "http://127.0.0.1:5173"})
            await asyncio.sleep(3)

            # ----------------------------------------------------
            # 1. Test Noise Sandbox Tab
            # ----------------------------------------------------
            print("Switching to Noise Sandbox Tab...")
            await send("Runtime.evaluate", {"params": {"expression": "document.querySelectorAll('nav button')[2].click()"}})
            await asyncio.sleep(2)

            click_noise = await send("Runtime.evaluate", {"params": {
                "expression": """
                    const btns = Array.from(document.querySelectorAll('button'));
                    const b = btns.find(x => x.textContent.includes('Run Simulation'));
                    if (b) { b.click(); 'clicked'; } else { 'not found'; }
                """,
                "returnByValue": True
            }})
            print("Noise Run Button Clicked:", click_noise.get("result", {}).get("result", {}).get("value"))
            await asyncio.sleep(5)

            sc_noise = await send("Page.captureScreenshot", {"params": {"format": "png"}})
            b64 = sc_noise["result"]["data"]
            with open("screenshots/11_noise_simulation_verified.png", "wb") as f:
                f.write(base64.b64decode(b64))
            print("Saved screenshots/11_noise_simulation_verified.png")

            # ----------------------------------------------------
            # 2. Test Benchmark Tab
            # ----------------------------------------------------
            print("\nSwitching to Benchmark Tab...")
            await send("Runtime.evaluate", {"params": {"expression": "document.querySelectorAll('nav button')[3].click()"}})
            await asyncio.sleep(2)

            click_bench = await send("Runtime.evaluate", {"params": {
                "expression": """
                    const btns = Array.from(document.querySelectorAll('button'));
                    const b = btns.find(x => x.textContent.includes('Run Full Benchmark'));
                    if (b) { b.click(); 'clicked'; } else { 'not found'; }
                """,
                "returnByValue": True
            }})
            print("Benchmark Run Button Clicked:", click_bench.get("result", {}).get("result", {}).get("value"))
            await asyncio.sleep(6)

            sc_bench = await send("Page.captureScreenshot", {"params": {"format": "png"}})
            b64_bench = sc_bench["result"]["data"]
            with open("screenshots/13_benchmark_completed_verified.png", "wb") as f:
                f.write(base64.b64decode(b64_bench))
            print("Saved screenshots/13_benchmark_completed_verified.png")

            # ----------------------------------------------------
            # 3. Test Generate Report
            # ----------------------------------------------------
            print("\nClicking Generate Report in Benchmark...")
            click_rep = await send("Runtime.evaluate", {"params": {
                "expression": """
                    const btns = Array.from(document.querySelectorAll('button'));
                    const b = btns.find(x => x.textContent.includes('Generate Report'));
                    if (b) { b.click(); 'clicked'; } else { 'not found'; }
                """,
                "returnByValue": True
            }})
            print("Report Button Clicked:", click_rep.get("result", {}).get("result", {}).get("value"))
            await asyncio.sleep(3)

            # Scroll to report
            await send("Runtime.evaluate", {"params": {"expression": "window.scrollTo(0, document.body.scrollHeight)"}})
            await asyncio.sleep(1)

            sc_rep = await send("Page.captureScreenshot", {"params": {"format": "png"}})
            b64_rep = sc_rep["result"]["data"]
            with open("screenshots/14_clinical_report_verified.png", "wb") as f:
                f.write(base64.b64decode(b64_rep))
            print("Saved screenshots/14_clinical_report_verified.png")

    finally:
        proc.terminate()
        print("Chrome finished.")

if __name__ == "__main__":
    asyncio.run(test())
