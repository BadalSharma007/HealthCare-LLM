import requests
import time

BASE_URL = "https://dev-30--medgemma-medical-medicalmodel-api.modal.run"

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
reset("test_user")

chat("test_user", "I have mild headache since morning")
chat("test_user", "Severe back pain for 3 weeks, can't walk")
chat("test_user", "Crushing chest pain and left arm is numb")
chat("test_user", "I have a cold")   # 4th message → session ended
