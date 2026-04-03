# ============================================================================
# This block imports Modal SDK and creates the Modal app.
# Modal is a cloud platform that lets you run Python functions on GPUs.
# "medgemma-medical" is the name of this app on your Modal dashboard.
# ============================================================================
import modal

app = modal.App("medgemma-medical")

# ============================================================================
# This block defines the Docker-like cloud environment (container image).
# It uses a slim Debian Linux base with Python 3.11 and installs the required
# ML libraries: torch (GPU compute), transformers (HuggingFace models),
# accelerate (multi-GPU support), bitsandbytes (4-bit quantization).
# This image is built once and cached on Modal's servers.
# ============================================================================
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "fastapi[standard]",
)

# ============================================================================
# This block creates a persistent key-value store on Modal cloud.
# It tracks how many messages each user has sent in their current session.
# "create_if_missing=True" means it auto-creates the store on first run.
# MAX_CHATS_PER_SESSION limits each user to 3 messages per session.
# ============================================================================
sessions = modal.Dict.from_name("medical-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 3

# ============================================================================
# This block maps health-related keywords to the correct medical specialist.
# When a user mentions "back pain", the system recommends an Orthopedic Surgeon.
# When a user mentions "chest", it maps to a Cardiologist — and so on.
# Used to auto-recommend the right doctor based on the user's complaint.
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

# ============================================================================
# This function scans the user's message for keywords from SPECIALIST_MAP.
# Returns the matching specialist name, or "General Physician" if no match.
# ============================================================================
def detect_specialist(text: str) -> str:
    lower = text.lower()
    for keyword, specialist in SPECIALIST_MAP.items():
        if keyword in lower:
            return specialist
    return "General Physician"

# ============================================================================
# These are the system prompts (instructions) given to the AI model.
# Each prompt tells the model HOW to behave for each severity level:
#
# LEVEL_1_PROMPT  → Mild issues: give general safe advice only
# LEVEL_2_PROMPT  → Moderate issues: ask questions + recommend specialist
# LEVEL_3_PROMPT  → Emergency/SOS: skip advice, direct to hospital immediately
# ============================================================================
LEVEL_1_PROMPT = """You are a friendly medical information assistant for Level 1 (mild concern).
The user has a general/mild health question.
- Give practical general health advice (rest, hydration, lifestyle tips).
- Do NOT prescribe any medicines or treatments.
- Keep it warm, simple and concise (2-3 sentences max).
- End by noting: if it persists, they should see a doctor."""

LEVEL_2_PROMPT = """You are a medical triage assistant for Level 2 (moderate concern).
The user has a moderate health concern.
- Ask 1-2 short clarifying questions to understand severity better (duration, location, intensity).
- Give general precautionary advice only (not prescriptions).
- Clearly recommend they consult a specific specialist (will be provided).
- Keep response concise."""

LEVEL_3_PROMPT = """You are an emergency medical assistant for Level 3 (serious/SOS).
The user has a serious or emergency health concern.
- Immediately tell them this is serious and they need urgent medical attention.
- Do NOT give advice — direct them to go to a hospital or call emergency services NOW.
- Be calm but very clear and urgent.
- One short sentence of what to do while waiting for help (if safe to give)."""

# ============================================================================
# This prompt is used to classify the severity of the user's message.
# The model is asked to respond with only LEVEL1, LEVEL2, or LEVEL3.
# This classification then decides which system prompt and response style to use.
# ============================================================================
SEVERITY_DETECT_PROMPT = """You are a medical severity classifier.
Given the user's health message, respond with ONLY one of: LEVEL1, LEVEL2, or LEVEL3.

LEVEL1 = Mild / general concern (cold, mild headache, tiredness, minor issue)
LEVEL2 = Moderate concern (ongoing pain, recurring symptoms, needs doctor visit)
LEVEL3 = Serious / emergency (severe chest pain, difficulty breathing, stroke signs, high fever in child, uncontrolled bleeding, suicidal thoughts)

User message: "{prompt}"
Severity:"""

# ============================================================================
# List of non-medical / personal questions to block.
# If the user asks any of these, the bot redirects them back to medical topics.
# ============================================================================
PERSONAL_KEYWORDS = ["your name", "who are you", "are you human", "do you feel", "your opinion", "how are you"]

# ============================================================================
# This block defines the main Modal class that runs on a GPU in the cloud.
# @app.cls        → deploys this class as a Modal service on an A10G GPU
# max_containers  → allows up to 100 parallel instances (100 simultaneous users)
# secrets         → securely loads HF_TOKEN from Modal's secret store
# @modal.concurrent → each container can handle up to 10 queued requests
# ============================================================================
@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=100,
)
@modal.concurrent(max_inputs=10)
class MedicalModel:

    # ========================================================================
    # This method runs ONCE when the container starts up on Modal's GPU server.
    # @modal.enter() tells Modal to run this before handling any requests.
    # It loads the MedGemma model in 4-bit quantized form (q4) to save GPU RAM.
    # The model and tokenizer are stored as self.model and self.tokenizer
    # so all other methods can reuse them without reloading.
    # ========================================================================
    @modal.enter()
    def load_model(self):
        import torch, os
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        token = os.environ["HF_TOKEN"]
        model_id = "google/medgemma-1.5-4b-it"

        # 4-bit quantization config — reduces model size from ~8GB to ~4GB on GPU
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

    # ========================================================================
    # Internal helper: builds the full prompt, runs the model, and returns
    # only the assistant's reply (strips the prompt from the output).
    # Used by chat() to generate responses for all 3 severity levels.
    # ========================================================================
    def _run_model(self, system_prompt: str, user_message: str, max_tokens: int = 250) -> str:
        full_prompt = f"<system>{system_prompt}</system>\nUser: {user_message}\nAssistant:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("Assistant:")[-1].strip()

    # ========================================================================
    # Internal helper: classifies the user's message into LEVEL1, LEVEL2, or LEVEL3.
    # Runs a short model inference (max 5 tokens) just to get the severity label.
    # This label then controls which response style and system prompt is used.
    # ========================================================================
    def _detect_severity(self, prompt: str) -> str:
        classify_prompt = SEVERITY_DETECT_PROMPT.format(prompt=prompt)
        inputs = self.tokenizer(classify_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True).upper()
        if "LEVEL3" in raw:
            return "LEVEL3"
        elif "LEVEL2" in raw:
            return "LEVEL2"
        return "LEVEL1"

    # ========================================================================
    # Main chat method — handles one message from a user.
    # Flow:
    #   1. Block personal/non-medical questions (guardrail)
    #   2. Check if user's session (3 chats) is exhausted
    #   3. Detect severity (LEVEL1 / LEVEL2 / LEVEL3)
    #   4. Detect which specialist to recommend
    #   5. Generate response based on severity level
    #   6. Increment session counter and return result
    # ========================================================================
    @modal.method()
    def chat(self, user_id: str, message: str) -> dict:
        import time

        # Block non-medical / personal questions
        if any(kw in message.lower() for kw in PERSONAL_KEYWORDS):
            return {
                "response": "I'm a medical information assistant. I can only help with health-related questions.",
                "level": "GUARDRAIL",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
            }

        # Check session limit — max 3 chats per session per user
        session = sessions.get(user_id, {"count": 0})
        count = session["count"]

        if count >= MAX_CHATS_PER_SESSION:
            return {
                "response": (
                    f"Your session of {MAX_CHATS_PER_SESSION} chats has ended. "
                    "Please start a new session to continue."
                ),
                "level": "SESSION_ENDED",
                "session_count": count,
                "session_ended": True,
                "specialist": None,
            }

        # Classify severity and detect the relevant specialist
        start = time.time()
        severity = self._detect_severity(message)
        specialist = detect_specialist(message)

        # Generate response — behaviour changes based on severity level
        if severity == "LEVEL1":
            # Mild: give general lifestyle/health tips, no prescriptions
            response = self._run_model(LEVEL_1_PROMPT, message)

        elif severity == "LEVEL2":
            # Moderate: ask clarifying questions + recommend specialist
            system = LEVEL_2_PROMPT + f"\nRecommend they see a: {specialist}"
            response = self._run_model(system, message)

        else:
            # SOS/Emergency: skip advice, direct to hospital + emergency number
            response = self._run_model(LEVEL_3_PROMPT, message)
            response += (
                f"\n\n🚨 EMERGENCY: Please call 112 (India) / 911 (US) immediately "
                f"or go to the nearest hospital. Ask for a {specialist}."
            )

        elapsed_ms = round((time.time() - start) * 1000)

        # Save updated session count back to Modal's persistent store
        new_count = count + 1
        sessions[user_id] = {"count": new_count}
        session_ended = new_count >= MAX_CHATS_PER_SESSION

        return {
            "response": response,
            "level": severity,
            "session_count": new_count,
            "session_ended": session_ended,
            "specialist": specialist,
            "time_ms": elapsed_ms,
        }

    # ========================================================================
    # Resets the session counter for a user back to 0.
    # Call this to allow the same user to start a fresh session of 3 chats.
    # ========================================================================
    @modal.method()
    def reset_session(self, user_id: str) -> dict:
        sessions[user_id] = {"count": 0}
        return {"status": "Session reset", "user_id": user_id}

    # ========================================================================
    # Single deployed HTTP endpoint — handles both chat and session reset.
    # Saves 1 endpoint slot (useful on Modal free tier: limit of 8 endpoints).
    #
    # For chat:  POST { "action": "chat",  "user_id": "...", "message": "..." }
    # For reset: POST { "action": "reset", "user_id": "..." }
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
# Local test entrypoint — only runs when you do "modal run app.py".
# NOT part of the deployed API. Used to test the full chat flow locally.
# Simulates a user sending 4 messages (3 valid + 1 that hits session limit).
# ============================================================================
@app.local_entrypoint()
def main():
    model = MedicalModel()
    user_id = "user_test_001"

    # Clear any previous session so the test starts fresh
    model.reset_session.remote(user_id)

    conversations = [
        "Mujhe Nigh me nind nhi aarhi 3 dino se..",           # likely LEVEL1
        "I've had severe back pain for 3 weeks, can't walk properly",  # likely LEVEL2
        "I'm having crushing chest pain and my left arm is numb",      # likely LEVEL3
        "I have a slight cold today",                      # 4th message → session ended
    ]

    for msg in conversations:
        print(f"\nUser [{user_id}]: {msg}")
        result = model.chat.remote(user_id, msg)

        print(f"Level     : {result['level']}")
        print(f"Specialist: {result['specialist']}")
        print(f"Session   : {result['session_count']}/{MAX_CHATS_PER_SESSION}")
        print(f"Response  : {result['response']}")
        if result.get("session_ended"):
            print(">> Session ended. User must start a new session.")
        print("-" * 70)
