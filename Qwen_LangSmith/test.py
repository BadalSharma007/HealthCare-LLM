import requests
import time

BASE_URL = "https://research-work90--qwen-langsmith-qwenmedical-api.modal.run"
MAX_CHATS = 3

def chat_api(user_id, message, image_path=""):
    import base64
    body = {"action": "chat", "user_id": user_id, "message": message}
    if image_path:
        with open(image_path, "rb") as f:
            body["image_base64"] = base64.b64encode(f.read()).decode()
    t_start = time.time()
    res = requests.post(BASE_URL, json=body, timeout=300)
    total_ms = round((time.time() - t_start) * 1000)
    data = res.json()
    data["_total_ms"] = total_ms
    return data

def reset_api(user_id):
    res = requests.post(BASE_URL, json={"action": "reset", "user_id": user_id}, timeout=60)
    return res.json()

def run_session():
    user_id = "interactive_user"

    print("\n" + "=" * 60)
    print("   Qwen2-VL + LangSmith Medical AI — Interactive Chat")
    print(f"   Session limit: {MAX_CHATS} messages")
    print("=" * 60)

    reset_api(user_id)
    print("Session started fresh.\n")

    pending_questions = []

    for chat_num in range(1, MAX_CHATS + 1):
        print(f"\n{'─' * 60}")
        print(f"  Chat {chat_num} of {MAX_CHATS}")
        print(f"{'─' * 60}")

        if pending_questions:
            print("\nThe AI has follow-up questions for you:")
            for i, q in enumerate(pending_questions, 1):
                print(f"  Q{i}: {q}")
            message = ""
            while not message:
                message = input("\nYour answer (answer both questions above): ").strip()
                if not message:
                    print("Please type something.")
        else:
            message = ""
            while not message:
                message = input("\nDescribe your symptom or health concern: ").strip()
                if not message:
                    print("Please type something.")

        image_path = input("Image path (leave blank to skip): ").strip().strip("'")

        print("\nProcessing... (please wait)")
        data = chat_api(user_id, message, image_path)

        print(f"\nLevel     : {data.get('level')}")
        print(f"Specialist: {data.get('specialist')}")
        print(f"Session   : {data.get('session_count')}/{MAX_CHATS}")
        print(f"Has Image : {data.get('has_image')}")
        print(f"\nResponse:\n{data.get('response')}")
        print(f"\nModel: {data.get('time_ms')}ms  |  Total: {data['_total_ms']}ms")

        if data.get("session_ended"):
            print("\n" + "=" * 60)
            print("  Session complete (3 chats used).")
            print("  Run the script again to start a new session.")
            print("=" * 60)
            break

        pending_questions = data.get("follow_up_questions", [])
        if pending_questions:
            print("\n[Next: answer the follow-up questions above]")

run_session()
