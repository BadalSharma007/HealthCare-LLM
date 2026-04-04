import requests
import time

# Apollo deployed API URL — update after running: modal deploy Apollo/app.py
BASE_URL = "https://dev-30--apollo-medical-apollomodel-api.modal.run"

def chat(user_id, message):
    t_start = time.time()
    res = requests.post(BASE_URL, json={"action": "chat", "user_id": user_id, "message": message})
    total_ms = round((time.time() - t_start) * 1000)
    data = res.json()
    print(f"Level     : {data.get('level')}")
    print(f"Specialist: {data.get('specialist')}")
    print(f"Session   : {data.get('session_count')}/3")
    print(f"Response  : {data.get('response')}")
    print(f"Model time: {data.get('time_ms')}ms  |  Total (network+model): {total_ms}ms")
    if data.get('session_ended'):
        print(">> Session ended. Reset to start again.")
    print("-" * 60)

def reset(user_id):
    res = requests.post(BASE_URL, json={"action": "reset", "user_id": user_id})
    print(f"Reset: {res.json()}")

# Reset session first
reset("apollo_test_user")

chat("apollo_test_user", "I have a mild headache since morning")
chat("apollo_test_user", "I've had severe back pain for 3 weeks, can't walk properly")
chat("apollo_test_user", "I'm having crushing chest pain and my left arm is numb")
chat("apollo_test_user", "I have a slight cold today")   # 4th → session ended
