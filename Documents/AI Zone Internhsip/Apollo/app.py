# ============================================================================
# Apollo Medical AI — app.py
# Model: FreedomIntelligence/Apollo-7B (7B parameters, open access)
# Separate from MedGemma — no shared code, no confusion between models.
# Deploy:  cd "AI Zone Internhsip" && modal deploy Apollo/app.py
# Test:    python3 Apollo/test.py
# ============================================================================
import modal

app = modal.App("apollo-medical")

# ============================================================================
# Container image — same libraries as MedGemma but for Apollo 7B model.
# boto3 for AWS S3 + DynamoDB storage.
# ============================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
        "boto3",
    )
    .add_local_file(
        local_path="Apollo/storage_handler.py",
        remote_path="/root/storage_handler.py"
    )
)

# ============================================================================
# Persistent session store for Apollo — separate from MedGemma sessions.
# Uses "apollo-sessions" — different from MedGemma's "medical-sessions".
# ============================================================================
sessions = modal.Dict.from_name("apollo-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 3

# ============================================================================
# Specialist keyword map — maps symptoms to correct medical specialist.
# ============================================================================
SPECIALIST_MAP = {
    "bone": "Orthopedic Surgeon",
    "back pain": "Orthopedic Surgeon",
    "spine": "Orthopedic Surgeon",
    "joint": "Orthopedic Surgeon",
    "fracture": "Orthopedic Surgeon",
    "heart": "Cardiologist",
    "chest pain": "Cardiologist",
    "chest": "Cardiologist",
    "brain": "Neurologist",
    "headache": "Neurologist",
    "migraine": "Neurologist",
    "seizure": "Neurologist",
    "skin": "Dermatologist",
    "rash": "Dermatologist",
    "acne": "Dermatologist",
    "eye": "Ophthalmologist",
    "vision": "Ophthalmologist",
    "anxiety": "Psychiatrist",
    "depression": "Psychiatrist",
    "mental": "Psychiatrist",
    "stress": "Psychiatrist",
    "child": "Pediatrician",
    "baby": "Pediatrician",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "cancer": "Oncologist",
    "tumor": "Oncologist",
    "kidney": "Nephrologist",
    "urine": "Nephrologist",
    "stomach": "Gastroenterologist",
    "liver": "Gastroenterologist",
    "digestion": "Gastroenterologist",
    "lung": "Pulmonologist",
    "breathing": "Pulmonologist",
    "asthma": "Pulmonologist",
    "ear": "ENT Specialist",
    "nose": "ENT Specialist",
    "throat": "ENT Specialist",
}

def detect_specialist(text: str) -> str:
    lower = text.lower()
    for keyword, specialist in SPECIALIST_MAP.items():
        if keyword in lower:
            return specialist
    return "General Physician"

# ============================================================================
# System prompts — same structured format as MedGemma for fair comparison.
# ============================================================================
LEVEL_1_PROMPT = """You are a medical information assistant. Reply in EXACTLY this format, nothing else:

Instruction: [one sentence telling the user what to do first]
General Advice: [2-3 practical tips like rest, hydration, lifestyle — NO medicines]
Consult: [{specialist}]
Keywords: [3-4 medical keywords from the user's complaint]
Follow-up Q1: [ask about duration — how long have they had this?]
Follow-up Q2: [ask about intensity — how severe is it on a scale?]"""

LEVEL_2_PROMPT = """You are a medical triage assistant. Reply in EXACTLY this format, nothing else:

Instruction: [one sentence — this needs attention, take it seriously]
General Advice: [2-3 precautionary tips — NO medicines or prescriptions]
Consult: [{specialist}]
Keywords: [3-4 medical keywords from the user's complaint]
Follow-up Q1: [ask about exact location of pain/symptom]
Follow-up Q2: [ask if symptoms are getting worse over time]"""

LEVEL_3_PROMPT = """You are an emergency medical assistant. Reply in EXACTLY this format, nothing else:

Instruction: [URGENT — go to hospital or call emergency services NOW]
General Advice: [one safe thing to do while waiting for help]
Consult: [{specialist}]
Keywords: [3-4 emergency medical keywords from the complaint]
Follow-up Q1: [ask if they are alone or someone is with them]
Follow-up Q2: [ask if symptoms started suddenly or gradually]"""

FOLLOWUP_PROMPT = """You are a medical assistant doing a follow-up.
The user's previous keywords were: {keywords}
Their answer to follow-up questions is: {message}

Reply in EXACTLY this format:
Instruction: [updated advice based on their answer]
General Advice: [more specific tips based on new information]
Consult: [{specialist}]
Keywords: [updated keywords including new information]
Follow-up Q1: [one more specific clarifying question]
Follow-up Q2: [ask about any other symptoms they may have]"""

# ============================================================================
# Severity classification prompt — same few-shot examples as MedGemma.
# Allows direct comparison of classification accuracy between both models.
# ============================================================================
SEVERITY_DETECT_PROMPT = """Classify the severity of this health message. Reply with ONLY one word.

Examples:
"I have a mild headache" -> LEVEL1
"I feel a bit tired" -> LEVEL1
"I have cold and runny nose" -> LEVEL1
"I have back pain for 2 weeks" -> LEVEL2
"I have recurring fever for 3 days" -> LEVEL2
"I feel dizzy when I stand up" -> LEVEL2
"I have crushing chest pain" -> LEVEL3
"I can't breathe" -> LEVEL3
"My left arm is numb and chest hurts" -> LEVEL3

Message: "{prompt}"
Answer (LEVEL1 or LEVEL2 or LEVEL3):"""

PERSONAL_KEYWORDS = ["your name", "who are you", "are you human", "do you feel", "your opinion", "how are you"]

# ============================================================================
# Apollo Modal class — deployed as a separate app on A10G GPU.
# App name: "apollo-medical" (different from "medgemma-medical")
# Apollo-7B is 7B params — fits on A10G with 4-bit NF4 quantization (~7GB).
# ============================================================================
@app.cls(
    image=image,
    gpu="A10G",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN
        modal.Secret.from_name("aws-secret"),          # AWS S3 + DynamoDB
    ],
    max_containers=100,
    # min_containers=1,  # uncomment when demoing — costs $1.10/hr on A10G
)
@modal.concurrent(max_inputs=10)
class ApolloModel:

    # ========================================================================
    # Loads Apollo-7B on container startup.
    # Apollo is open access — no HF token gating needed (but passed anyway).
    # 4-bit NF4 quantization: 14GB model → ~7GB on GPU.
    # Apollo uses ChatML prompt format (different from MedGemma).
    # ========================================================================
    @modal.enter()
    def load_model(self):
        import torch, os, sys
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        token = os.environ.get("HF_TOKEN", "")
        model_id = "FreedomIntelligence/Apollo-7B"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=token
        )

        sys.path.insert(0, "/root")
        from storage_handler import store_output_to_dynamodb, store_output_log_to_s3
        self.store_dynamodb = store_output_to_dynamodb
        self.store_s3       = store_output_log_to_s3

    # ========================================================================
    # Apollo uses ChatML format — different from MedGemma's <system> tags.
    # <|im_start|>system ... <|im_end|>
    # <|im_start|>user   ... <|im_end|>
    # <|im_start|>assistant
    # ========================================================================
    def _run_model(self, system_prompt: str, user_message: str, max_tokens: int = 250) -> str:
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = result.replace("```", "").strip()
        for stop in ["\nUser:", "\n<|im_start|>", "<|im_end|>", "User:"]:
            if stop in result:
                result = result.split(stop)[0].strip()
        return result

    # ========================================================================
    # Classifies user message into LEVEL1, LEVEL2, or LEVEL3.
    # Same logic as MedGemma for fair accuracy comparison.
    # ========================================================================
    def _detect_severity(self, prompt: str) -> str:
        classify_prompt = SEVERITY_DETECT_PROMPT.format(prompt=prompt)
        inputs = self.tokenizer(classify_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).upper().strip()
        if "LEVEL3" in raw:
            return "LEVEL3"
        elif "LEVEL2" in raw:
            return "LEVEL2"
        return "LEVEL1"

    # ========================================================================
    # Main chat method — same flow as MedGemma for fair comparison.
    # ========================================================================
    @modal.method()
    def chat(self, user_id: str, message: str) -> dict:
        import time

        if any(kw in message.lower() for kw in PERSONAL_KEYWORDS):
            return {
                "response": "I'm a medical information assistant. I can only help with health-related questions.",
                "level": "GUARDRAIL",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
            }

        session = sessions.get(user_id, {"count": 0})
        count = session["count"]

        if count >= MAX_CHATS_PER_SESSION:
            return {
                "response": f"Your session of {MAX_CHATS_PER_SESSION} chats has ended. Please start a new session.",
                "level": "SESSION_ENDED",
                "session_count": count,
                "session_ended": True,
                "specialist": None,
            }

        start = time.time()
        severity = self._detect_severity(message)
        specialist = detect_specialist(message)

        prev_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": ""})
        prev_keywords = prev_data.get("keywords", "")

        if count == 0:
            if severity == "LEVEL1":
                system = LEVEL_1_PROMPT.replace("{specialist}", specialist)
            elif severity == "LEVEL2":
                system = LEVEL_2_PROMPT.replace("{specialist}", specialist)
            else:
                system = LEVEL_3_PROMPT.replace("{specialist}", specialist)
            response = self._run_model(system, message)
        else:
            system = FOLLOWUP_PROMPT.replace("{keywords}", prev_keywords or "not available") \
                                    .replace("{specialist}", specialist) \
                                    .replace("{message}", message)
            response = self._run_model(system, message)

        keywords = ""
        for line in response.split("\n"):
            if line.strip().startswith("Keywords:"):
                keywords = line.replace("Keywords:", "").strip()
                break

        if severity == "LEVEL3":
            response += (
                f"\n\n🚨 EMERGENCY: Please call 112 (India) / 911 (US) immediately "
                f"or go to the nearest hospital. Ask for a {specialist}."
            )

        elapsed_ms = round((time.time() - start) * 1000)
        new_count = count + 1
        sessions[user_id] = {"count": new_count}
        session_ended = new_count >= MAX_CHATS_PER_SESSION

        chat_id = f"chat_{user_id}"
        session_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": ""})
        session_data["keywords"] = keywords
        query_key = f"query_{new_count}"
        session_data["chats"][query_key] = {
            "input":  {"query_text": message},
            "output": {"text": response}
        }

        record = {
            "user_id":            user_id,
            "chat_id":            chat_id,
            "chats":              session_data["chats"],
            "severity":           severity.lower(),
            "doctor_specialist":  specialist,
            "recommended_action": "consult immediately" if severity == "LEVEL3" else "see a specialist" if severity == "LEVEL2" else "monitor symptoms",
            "timestamp":          __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "language":           "en",
            "model":              "Apollo-7B",
        }

        sessions[f"{user_id}_record"] = session_data
        self.store_dynamodb(record)
        self.store_s3(record)

        return {
            "response": response,
            "level": severity,
            "session_count": new_count,
            "session_ended": session_ended,
            "specialist": specialist,
            "time_ms": elapsed_ms,
        }

    @modal.method()
    def reset_session(self, user_id: str) -> dict:
        sessions[user_id] = {"count": 0}
        sessions[f"{user_id}_record"] = {"chats": {}, "keywords": ""}
        return {"status": "Session reset", "user_id": user_id}

    # ========================================================================
    # Single HTTP endpoint — same structure as MedGemma for easy comparison.
    # POST { "action": "chat",  "user_id": "...", "message": "..." }
    # POST { "action": "reset", "user_id": "..." }
    # ========================================================================
    @modal.fastapi_endpoint(method="POST")
    def api(self, body: dict) -> dict:
        action  = body.get("action", "chat")
        user_id = body.get("user_id", "anonymous")
        if action == "reset":
            return self.reset_session.local(user_id)
        message = body.get("message", "")
        if not message:
            return {"error": "message is required"}
        return self.chat.local(user_id, message)


# ============================================================================
# Local test entrypoint — run with: modal run Apollo/app.py
# Uses same 4 prompts as MedGemma test for direct comparison.
# ============================================================================
@app.local_entrypoint()
def main():
    model = ApolloModel()
    user_id = "apollo_test_001"

    model.reset_session.remote(user_id)

    conversations = [
        "I have a mild headache since morning",
        "I've had severe back pain for 3 weeks, can't walk properly",
        "I'm having crushing chest pain and my left arm is numb",
        "I have a slight cold today",   # 4th → session ended
    ]

    for msg in conversations:
        print(f"\nUser [{user_id}]: {msg}")
        result = model.chat.remote(user_id, msg)
        print(f"Level     : {result['level']}")
        print(f"Specialist: {result['specialist']}")
        print(f"Session   : {result['session_count']}/{MAX_CHATS_PER_SESSION}")
        print(f"Response  : {result['response']}")
        print(f"Time      : {result.get('time_ms')}ms")
        if result.get("session_ended"):
            print(">> Session ended.")
        print("-" * 70)
