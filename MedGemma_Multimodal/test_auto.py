"""
Automated test for MedGemma Multimodal medical chatbot.
Sends 3 hardcoded messages and prints full responses.
No interactive input — fully automated.
"""
import requests
import json

BASE_URL = "https://research-work90--medgemma-multimodal-medgemmamultimodal-api.modal.run"
USER_ID = "auto_test_neck_001"

MESSAGES = [
    "I have neck pain",
    "2 months, severity 6, gets worse",
    "Also have headaches",
]

def post(action, **kwargs):
    body = {"action": action, "user_id": USER_ID, **kwargs}
    resp = requests.post(BASE_URL, json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()

def main():
    print("=" * 70)
    print("MedGemma Multimodal Auto Test — neck pain / follow-ups")
    print(f"URL: {BASE_URL}")
    print("=" * 70)

    # Reset session so we start fresh
    reset = post("reset")
    print(f"\n[RESET] {reset}\n")

    for i, msg in enumerate(MESSAGES, 1):
        print(f"--- Chat {i} ---")
        print(f"User: {msg}")
        result = post("chat", message=msg)
        print(f"Level     : {result.get('level')}")
        print(f"Specialist: {result.get('specialist')}")
        print(f"Session   : {result.get('session_count')}/{3}")
        print(f"Time      : {result.get('time_ms')}ms")
        print(f"Response  :\n{result.get('response')}")
        if result.get("session_ended"):
            print(">> Session ended.")
        print()

if __name__ == "__main__":
    main()
