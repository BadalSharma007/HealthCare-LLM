"""
Automated test for Qwen2-VL + LangSmith Medical AI chatbot.
Sends 3 hardcoded messages and prints full responses.
No interactive input — fully automated.
"""
import requests
import time

BASE_URL = "https://research-work90--qwen-langsmith-qwenmedical-api.modal.run"
USER_ID = "auto_test_qwen_001"

MESSAGES = [
    "I have neck pain",
    "2 months, severity 6, gets worse when I look down",
    "Also have headaches and tingling in my fingers",
]

def post(action, **kwargs):
    body = {"action": action, "user_id": USER_ID, **kwargs}
    t = time.time()
    resp = requests.post(BASE_URL, json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    data["_network_ms"] = round((time.time() - t) * 1000)
    return data

def main():
    print("=" * 70)
    print("Qwen2-VL + LangSmith Medical AI — Auto Test")
    print(f"URL: {BASE_URL}")
    print("=" * 70)

    reset = post("reset")
    print(f"\n[RESET] {reset}\n")

    session_start = time.time()
    for i, msg in enumerate(MESSAGES, 1):
        print(f"--- Chat {i} ---")
        print(f"User: {msg}")
        result = post("chat", message=msg)
        model_ms   = result.get("time_ms", 0)
        network_ms = result.get("_network_ms", 0)
        overhead   = network_ms - model_ms
        print(f"Level     : {result.get('level')}")
        print(f"Specialist: {result.get('specialist')}")
        print(f"Session   : {result.get('session_count')}/3")
        print(f"Model     : {model_ms}ms")
        print(f"Network   : {network_ms}ms  (overhead: {overhead}ms)")
        print(f"Response  :\n{result.get('response')}")
        if result.get("session_ended"):
            print(">> Session ended.")
        print()

    total = round((time.time() - session_start) * 1000)
    print(f"Total session time: {total}ms ({total//1000}s)")

if __name__ == "__main__":
    main()
