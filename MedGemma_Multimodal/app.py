# ============================================================================
# MedGemma Multimodal Medical AI — app.py
# Model: google/medgemma-1.5-4b-it  (text + image)
# Deploy:  cd "AI Zone Internhsip" && modal deploy MedGemma_Multimodal/app.py
# Test:    python3 MedGemma_Multimodal/test.py
# ============================================================================
import modal

app = modal.App("medgemma-multimodal")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
        "boto3",
        "Pillow",          # image decoding for multimodal input
    )
    .add_local_file(
        local_path="MedGemma_Multimodal/storage_handler.py",
        remote_path="/root/storage_handler.py"
    )
)

sessions = modal.Dict.from_name("medgemma-mm-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 3

# ============================================================================
# Expanded specialist map — ordered list so multi-word phrases match before
# single words (e.g. "chest pain" before "chest", "pilonidal sinus" before "sinus").
# ============================================================================
SPECIALIST_MAP = [
    # ── Specific conditions first ────────────────────────────────────────────
    ("pilonidal sinus",     "General Surgeon"),
    ("pilonidal",           "General Surgeon"),
    ("appendicitis",        "General Surgeon"),
    ("gallstone",           "General Surgeon"),
    ("hernia",              "General Surgeon"),
    ("appendix",            "General Surgeon"),
    ("varicose",            "General Surgeon"),
    ("fistula",             "General Surgeon"),
    ("hemorrhoid",          "General Surgeon"),
    ("piles",               "General Surgeon"),
    ("rectal",              "General Surgeon"),
    ("anal",                "General Surgeon"),
    ("blood in cough",      "Pulmonologist"),
    ("coughing blood",      "Pulmonologist"),
    ("cough with blood",    "Pulmonologist"),
    ("can't breathe",       "Pulmonologist"),
    ("cannot breathe",      "Pulmonologist"),
    ("shortness of breath", "Pulmonologist"),
    ("chest pain",          "Cardiologist"),
    ("left arm",            "Cardiologist"),
    ("heart attack",        "Cardiologist"),
    ("heart failure",       "Cardiologist"),
    ("back pain",           "Orthopedic Surgeon"),
    ("neck pain",           "Orthopedic Surgeon"),
    ("knee pain",           "Orthopedic Surgeon"),
    ("shoulder pain",       "Orthopedic Surgeon"),
    ("joint pain",          "Orthopedic Surgeon"),
    ("leg pain",            "Orthopedic Surgeon"),
    ("foot pain",           "Orthopedic Surgeon"),
    ("arm pain",            "Orthopedic Surgeon"),
    ("weight gain",         "Endocrinologist"),
    ("weight loss",         "Endocrinologist"),
    ("running nose",        "ENT Specialist"),
    ("runny nose",          "ENT Specialist"),
    ("sore throat",         "ENT Specialist"),
    ("hair loss",           "Dermatologist"),
    ("blood pressure",      "Cardiologist"),
    ("high bp",             "Cardiologist"),
    ("blurred vision",      "Ophthalmologist"),
    ("double vision",       "Ophthalmologist"),
    ("abdominal pain",      "Gastroenterologist"),
    ("stomach pain",        "Gastroenterologist"),
    ("stomach ache",        "Gastroenterologist"),
    ("urinary tract",       "Urologist"),
    ("blood in urine",      "Urologist"),
    ("kidney stone",        "Urologist"),
    ("irritable bowel",     "Gastroenterologist"),
    ("fatty liver",         "Gastroenterologist"),
    ("acid reflux",         "Gastroenterologist"),
    ("panic attack",        "Psychiatrist"),
    ("mood swing",          "Psychiatrist"),
    # ── Single keyword fallbacks ─────────────────────────────────────────────
    ("shoulder",    "Orthopedic Surgeon"),
    ("knee",        "Orthopedic Surgeon"),
    ("spine",       "Orthopedic Surgeon"),
    ("bone",        "Orthopedic Surgeon"),
    ("fracture",    "Orthopedic Surgeon"),
    ("joint",       "Orthopedic Surgeon"),
    ("hip",         "Orthopedic Surgeon"),
    ("wrist",       "Orthopedic Surgeon"),
    ("elbow",       "Orthopedic Surgeon"),
    ("ankle",       "Orthopedic Surgeon"),
    ("muscle",      "Orthopedic Surgeon"),
    ("tendon",      "Orthopedic Surgeon"),
    ("ligament",    "Orthopedic Surgeon"),
    ("neck",        "Orthopedic Surgeon"),
    ("heart",       "Cardiologist"),
    ("chest",       "Cardiologist"),
    ("palpitation", "Cardiologist"),
    ("cardiac",     "Cardiologist"),
    ("brain",       "Neurologist"),
    ("headache",    "Neurologist"),
    ("migraine",    "Neurologist"),
    ("seizure",     "Neurologist"),
    ("numbness",    "Neurologist"),
    ("tingling",    "Neurologist"),
    ("dizziness",   "Neurologist"),
    ("stroke",      "Neurologist"),
    ("tremor",      "Neurologist"),
    ("memory",      "Neurologist"),
    ("skin",        "Dermatologist"),
    ("rash",        "Dermatologist"),
    ("acne",        "Dermatologist"),
    ("eczema",      "Dermatologist"),
    ("itching",     "Dermatologist"),
    ("hives",       "Dermatologist"),
    ("psoriasis",   "Dermatologist"),
    ("eye",         "Ophthalmologist"),
    ("vision",      "Ophthalmologist"),
    ("anxiety",     "Psychiatrist"),
    ("depression",  "Psychiatrist"),
    ("mental",      "Psychiatrist"),
    ("stress",      "Psychiatrist"),
    ("insomnia",    "Psychiatrist"),
    ("panic",       "Psychiatrist"),
    ("sleep",       "Psychiatrist"),
    ("child",       "Pediatrician"),
    ("baby",        "Pediatrician"),
    ("infant",      "Pediatrician"),
    ("toddler",     "Pediatrician"),
    ("diabetes",    "Endocrinologist"),
    ("thyroid",     "Endocrinologist"),
    ("hormonal",    "Endocrinologist"),
    ("cancer",      "Oncologist"),
    ("tumor",       "Oncologist"),
    ("lump",        "Oncologist"),
    ("kidney",      "Nephrologist"),
    ("urine",       "Nephrologist"),
    ("renal",       "Nephrologist"),
    ("stomach",     "Gastroenterologist"),
    ("liver",       "Gastroenterologist"),
    ("digestion",   "Gastroenterologist"),
    ("vomiting",    "Gastroenterologist"),
    ("nausea",      "Gastroenterologist"),
    ("diarrhea",    "Gastroenterologist"),
    ("constipation","Gastroenterologist"),
    ("abdomen",     "Gastroenterologist"),
    ("bowel",       "Gastroenterologist"),
    ("lung",        "Pulmonologist"),
    ("breathing",   "Pulmonologist"),
    ("asthma",      "Pulmonologist"),
    ("cough",       "Pulmonologist"),
    ("breath",      "Pulmonologist"),
    ("pneumonia",   "Pulmonologist"),
    ("ear",         "ENT Specialist"),
    ("nose",        "ENT Specialist"),
    ("throat",      "ENT Specialist"),
    ("sinus",       "ENT Specialist"),   # after pilonidal sinus above
    ("tonsil",      "ENT Specialist"),
    ("hearing",     "ENT Specialist"),
    ("sneezing",    "ENT Specialist"),
    ("urinary",     "Urologist"),
    ("bladder",     "Urologist"),
    ("prostate",    "Urologist"),
    ("period",      "Gynecologist"),
    ("menstrual",   "Gynecologist"),
    ("pregnancy",   "Gynecologist"),
    ("ovary",       "Gynecologist"),
    ("tooth",       "Dentist"),
    ("teeth",       "Dentist"),
    ("gum",         "Dentist"),
    ("dental",      "Dentist"),
]

def detect_specialist(text: str) -> str:
    lower = text.lower()
    for keyword, specialist in SPECIALIST_MAP:
        if keyword in lower:
            return specialist
    return "General Physician"

# ============================================================================
# Medicine names to strip from advice — model sometimes recommends them.
# ============================================================================
MEDICINE_WORDS = [
    "ibuprofen", "paracetamol", "aspirin", "acetaminophen", "naproxen",
    "antibiotic", "steroid", "prescription", "tablet", "pill", "medication",
    "drug", "dose", "mg", "tylenol", "advil", "motrin", "aleve",
]

def strip_medicines(text: str) -> str:
    parts = [p.strip() for p in text.replace(";", "|").split("|")]
    clean = [p for p in parts if not any(m in p.lower() for m in MEDICINE_WORDS)]
    return "; ".join(clean) if clean else "Rest and avoid strenuous activity; stay well hydrated"

# ============================================================================
# Chat 1 prompts — one-shot example, model imitates the format.
# ============================================================================
LEVEL_1_PROMPT = """You are a medical information assistant.

Format example only — do not copy this content:
- I have fever since yesterday
- Mild Concern
- Instruction: Rest at home, drink plenty of fluids, and monitor your temperature every few hours.
- General Advice: Stay hydrated with water and electrolytes; rest in a cool room; use a cold compress on your forehead.
- Consult: [General Physician]
- Keywords: fever, dehydration, viral infection, temperature
- Q1: How high is your temperature — have you measured it with a thermometer?
- Q2: How long have you had the fever — did it start hours ago or a few days ago?

Now write a response for this patient using the same format, with content specific to their complaint. STOP after Q2. Do not write Q3.
Patient complaint: {user_message}
Specialist: {specialist}"""

LEVEL_2_PROMPT = """You are a medical triage assistant.

Format example only — do not copy this content:
- I have knee pain when climbing stairs
- Moderate Concern
- Instruction: Avoid putting weight on the knee and visit a doctor within 1-2 days for proper evaluation.
- General Advice: Apply ice wrapped in cloth for 15 minutes at a time; keep the leg elevated when resting; avoid bending the knee deeply.
- Consult: [Orthopedic Surgeon]
- Keywords: knee pain, joint inflammation, cartilage, mobility
- Q1: Is the pain on the inner side, outer side, or front of your knee?
- Q2: Has the pain been getting worse over the past few days or weeks?

Now write a response for this patient using the same format, with content specific to their complaint. STOP after Q2. Do not write Q3.
Patient complaint: {user_message}
Specialist: {specialist}"""

LEVEL_3_PROMPT = """You are an emergency medical assistant.

Format example only — do not copy this content:
- I am having difficulty breathing and my lips are turning blue
- EMERGENCY
- Instruction: Call 112 or 911 immediately and do not wait — this requires emergency care right now.
- General Advice: Sit upright and stay as calm as possible while waiting for help to arrive.
- Consult: [Pulmonologist]
- Keywords: respiratory distress, hypoxia, emergency, cyanosis
- Q1: Is there anyone with you right now who can call for help?
- Q2: Did the breathing difficulty come on suddenly or has it been building up gradually?

Now write a response for this patient using the same format, with content specific to their complaint. STOP after Q2. Do not write Q3.
Patient complaint: {user_message}
Specialist: {specialist}"""

# ============================================================================
# Image analysis prompt — used when user sends an image.
# Model describes what it sees medically, then the chat flow continues normally.
# ============================================================================
IMAGE_ANALYSIS_PROMPT = """You are a medical image analysis assistant. The patient has shared a medical image.
Describe what you observe in the image in simple terms a patient can understand.
Focus on: visible symptoms, affected area, any visible abnormality.
Keep it brief — 2-3 sentences. Do NOT suggest diagnosis or medicines.
Image observation:"""

# ============================================================================
# Follow-up prompts (Chat 2/3) — 3 focused calls, Python assembles the format.
# ============================================================================
FOLLOWUP_ADVICE_PROMPT = """You are a {specialist} giving lifestyle advice to a patient with {keywords}.
Patient says: {message}

Write 2 short practical tips to help the patient manage {keywords} at home. No medicine names. Use semicolons between tips.
Tips:"""

FOLLOWUP_Q1_PROMPT = """You are a {specialist} asking a follow-up question to a patient with {keywords}.
The patient said: {message}

Ask the patient one specific question to better understand their {keywords} condition.
Question:"""

FOLLOWUP_Q2_PROMPT = """You are a {specialist} asking about other symptoms to a patient with {keywords}.
The patient said: {message}

Ask the patient one question about any other symptoms they may have noticed.
Question:"""

PERSONAL_KEYWORDS = ["your name", "who are you", "are you human", "do you feel", "your opinion", "how are you"]

# ============================================================================
# Hardcoded emergency follow-up — never rely on model for LEVEL3 follow-up.
# ============================================================================
EMERGENCY_FOLLOWUP = {
    "Cardiologist":      ("Do not move — sit or lie down quietly and loosen any tight clothing.", "Is there someone with you who can call 112 or 911 right now?", "Are you feeling increasing chest pressure or spreading pain to your jaw or arm?"),
    "Pulmonologist":     ("Sit upright, open windows for fresh air, and do not lie flat.", "Is someone with you who can call emergency services immediately?", "Is the difficulty breathing getting worse every few minutes?"),
    "Neurologist":       ("Do not give the patient anything to eat or drink — keep them still and calm.", "Is someone with you right now to call 112 or 911?", "Are symptoms like numbness or vision loss getting worse rapidly?"),
    "General Physician": ("Stay calm, sit down, and do not attempt to drive yourself to hospital.", "Is there someone nearby who can call emergency services for you?", "Are any of your symptoms rapidly getting worse right now?"),
}

def get_emergency_followup(specialist: str, keywords: str) -> tuple:
    advice, q1, q2 = EMERGENCY_FOLLOWUP.get(specialist, EMERGENCY_FOLLOWUP["General Physician"])
    return advice, q1, q2


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    max_containers=100,
    # min_containers=1,  # uncomment when demoing — costs $1.10/hr on A10G
)
@modal.concurrent(max_inputs=10)
class MedGemmaMultimodal:

    @modal.enter()
    def load_model(self):
        import torch, os, sys
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        token = os.environ.get("HF_TOKEN", "")
        model_id = "google/medgemma-1.5-4b-it"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # AutoProcessor handles both text tokenization and image preprocessing
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=token
        )

        sys.path.insert(0, "/root")
        from storage_handler import store_output_to_dynamodb, store_output_log_to_s3
        self.store_dynamodb = store_output_to_dynamodb
        self.store_s3       = store_output_log_to_s3

    def _run_model(self, system_prompt: str, user_message: str,
                   max_tokens: int = 250, primer: str = "", pil_image=None) -> str:
        """
        Run the model with optional image input.
        primer: text prepended to the model's response (forces completion style).
        pil_image: PIL.Image.Image object for multimodal input.
        """
        # Build message content
        if pil_image is not None:
            content = [
                {"type": "image"},
                {"type": "text", "text": f"{system_prompt}\n\n{user_message}"},
            ]
        else:
            content = [{"type": "text", "text": f"{system_prompt}\n\n{user_message}"}]

        messages = [{"role": "user", "content": content}]

        # Apply Gemma chat template — adds <start_of_turn>model\n at the end
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if primer:
            text += primer

        # Process inputs (image-aware)
        if pil_image is not None:
            inputs = self.processor(
                text=text, images=pil_image, return_tensors="pt"
            ).to("cuda")
        else:
            inputs = self.processor(text=text, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(new_tokens, skip_special_tokens=True)
        result = result.replace("```", "").strip()

        # Stop tokens — prevent echoing prompt or generating extra turns
        for stop in ["<end_of_turn>", "<start_of_turn>", "\nUser:", "User:",
                     "\nNow write", "\nPatient complaint", "\n- Q3:", "\nQ3:"]:
            if stop in result:
                result = result.split(stop)[0].strip()
        return result

    def _detect_severity(self, prompt: str) -> str:
        classify_prompt = """Classify the severity of this health message. Reply with ONLY one word.

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
"cough with blood" -> LEVEL3
"I cannot breathe" -> LEVEL3
"I am unconscious" -> LEVEL3
"condition is so bad" -> LEVEL2
"the pain is very bad" -> LEVEL2
"it is getting worse" -> LEVEL2
"I feel terrible" -> LEVEL2
"nothing is helping" -> LEVEL2

Message: "{prompt}"
Answer (LEVEL1 or LEVEL2 or LEVEL3):""".format(prompt=prompt)

        messages = [{"role": "user", "content": [{"type": "text", "text": classify_prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True).upper().strip()
        if "LEVEL3" in raw:
            return "LEVEL3"
        elif "LEVEL2" in raw:
            return "LEVEL2"
        return "LEVEL1"

    @modal.method()
    def chat(self, user_id: str, message: str, image_b64: str = "") -> dict:
        import time, base64, io

        if any(kw in message.lower() for kw in PERSONAL_KEYWORDS):
            return {
                "response": "I'm a medical information assistant. I can only help with health-related questions.",
                "level": "GUARDRAIL",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
                "follow_up_questions": [],
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
                "follow_up_questions": [],
            }

        # Decode image if provided
        pil_image = None
        image_observation = ""
        if image_b64:
            try:
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                # Run image analysis first — short description, no diagnosis
                image_observation = self._run_model(
                    IMAGE_ANALYSIS_PROMPT, message,
                    max_tokens=100, pil_image=pil_image
                )
                # Append observation to message for severity + specialist detection
                message = f"{message} [Image: {image_observation}]"
            except Exception:
                pil_image = None  # silently fall back to text-only on decode error

        start = time.time()
        severity = self._detect_severity(message)

        prev_data       = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": ""})
        prev_keywords   = prev_data.get("keywords", "")
        prev_specialist = prev_data.get("specialist", "")

        # Chat 1: detect from message. Chat 2/3: always reuse Chat 1's specialist.
        if count == 0:
            specialist = detect_specialist(message)
        else:
            specialist = prev_specialist or detect_specialist(prev_keywords) or detect_specialist(message)

        # ----------------------------------------------------------------
        # CHAT 1 — one-shot example prompt → model outputs full dash format
        # ----------------------------------------------------------------
        if count == 0:
            if severity == "LEVEL1":
                system = LEVEL_1_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message)
            elif severity == "LEVEL2":
                system = LEVEL_2_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message)
            else:
                system = LEVEL_3_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message)

            raw = self._run_model(system, message, pil_image=pil_image)

            # If model echoed system prompt, extract only the dash-formatted block
            if not raw.strip().startswith("- "):
                lines = raw.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("- "):
                        raw = "\n".join(lines[i:])
                        break

            # Strip medicine names from General Advice line
            cleaned_lines = []
            for line in raw.split("\n"):
                s = line.strip().lstrip("- ")
                if s.startswith("General Advice:"):
                    advice_text = s.replace("General Advice:", "").strip()
                    cleaned_lines.append(f"- General Advice: {strip_medicines(advice_text)}")
                else:
                    cleaned_lines.append(line)
            response = "\n".join(cleaned_lines)

            # Prepend image observation if available
            if image_observation:
                response = f"- Image Observation: {image_observation}\n" + response

        # ----------------------------------------------------------------
        # CHAT 2/3 — completion-style prompt, Python assembles dash format
        # ----------------------------------------------------------------
        else:
            kw = prev_keywords or "your symptoms"

            if severity == "LEVEL3":
                advice, q1, q2 = get_emergency_followup(specialist, kw)
                response = (
                    f"- {kw}: {specialist}\n"
                    f"- EMERGENCY: Call 112 (India) or 911 (US) NOW or go to the nearest hospital.\n"
                    f"- Immediate Action: {advice}\n"
                    f"- Q1: {q1}\n"
                    f"- Q2: {q2}"
                )
            else:
                def fmt(prompt, **kwargs):
                    for k, v in kwargs.items():
                        prompt = prompt.replace("{" + k + "}", v)
                    return prompt

                advice_raw = self._run_model(
                    fmt(FOLLOWUP_ADVICE_PROMPT, specialist=specialist, keywords=kw, message=message),
                    message, max_tokens=80, primer="Tips:", pil_image=pil_image
                )
                advice_text = strip_medicines(advice_raw.split("\n")[0].strip())
                if not advice_text:
                    advice_text = "Rest and avoid strenuous activity; stay well hydrated"

                q1_raw = self._run_model(
                    fmt(FOLLOWUP_Q1_PROMPT, specialist=specialist, keywords=kw, message=message),
                    message, max_tokens=60, primer="Question:"
                )
                q1_text = q1_raw.split("\n")[0].strip()
                if not q1_text:
                    q1_text = "Can you describe how your symptom has changed since it started?"

                q2_raw = self._run_model(
                    fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw, message=message),
                    message, max_tokens=60, primer="Question:"
                )
                q2_text = q2_raw.split("\n")[0].strip()
                if not q2_text:
                    q2_text = "Have you noticed any other symptoms appearing alongside this?"

                response = (
                    f"- {kw}: {specialist} + {advice_text}\n"
                    f"- Q1: {q1_text}\n"
                    f"- Q2: {q2_text}"
                )

                if image_observation:
                    response = f"- Image Observation: {image_observation}\n" + response

        # ----------------------------------------------------------------
        # Extract keywords and follow-up questions from assembled response
        # ----------------------------------------------------------------
        keywords = ""
        follow_up_questions = []
        for line in response.split("\n"):
            s = line.strip().lstrip("- ")
            if s.startswith("Keywords:"):
                keywords = s.replace("Keywords:", "").strip()
            elif s.upper().startswith("Q1:"):
                follow_up_questions.append(s[3:].strip())
            elif s.upper().startswith("Q2:"):
                follow_up_questions.append(s[3:].strip())

        if not keywords:
            keywords = prev_keywords

        # Emergency banner for Chat 1 LEVEL3
        if severity == "LEVEL3" and count == 0:
            response += (
                f"\n\nEMERGENCY: Please call 112 (India) / 911 (US) immediately "
                f"or go to the nearest hospital. Ask for a {specialist}."
            )

        elapsed_ms = round((time.time() - start) * 1000)
        new_count  = count + 1
        sessions[user_id] = {"count": new_count}
        session_ended = new_count >= MAX_CHATS_PER_SESSION

        chat_id      = f"chat_{user_id}"
        session_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": ""})
        session_data["keywords"]   = keywords
        session_data["specialist"] = specialist
        session_data["chats"][f"query_{new_count}"] = {
            "input":  {"query_text": message, "has_image": bool(image_b64)},
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
            "model":              "MedGemma-1.5-4b-it",
        }

        sessions[f"{user_id}_record"] = session_data
        self.store_dynamodb(record)
        self.store_s3(record)

        return {
            "response":            response,
            "level":               severity,
            "session_count":       new_count,
            "session_ended":       session_ended,
            "specialist":          specialist,
            "follow_up_questions": follow_up_questions,
            "time_ms":             elapsed_ms,
            "has_image":           bool(image_b64),
        }

    @modal.method()
    def reset_session(self, user_id: str) -> dict:
        sessions[user_id] = {"count": 0}
        sessions[f"{user_id}_record"] = {"chats": {}, "keywords": "", "specialist": ""}
        return {"status": "Session reset", "user_id": user_id}

    @modal.fastapi_endpoint(method="POST")
    def api(self, body: dict) -> dict:
        action  = body.get("action", "chat")
        user_id = body.get("user_id", "anonymous")
        if action == "reset":
            return self.reset_session.local(user_id)
        message = body.get("message", "")
        if not message:
            return {"error": "message is required"}
        image_b64 = body.get("image_base64", "")
        return self.chat.local(user_id, message, image_b64)


@app.local_entrypoint()
def main():
    model = MedGemmaMultimodal()
    user_id = "mm_test_001"
    model.reset_session.remote(user_id)

    conversations = [
        ("I have a mild headache since morning", ""),
        ("I have severe back pain for 3 weeks, can't walk properly", ""),
        ("I'm having crushing chest pain and my left arm is numb", ""),
    ]

    for msg, img_b64 in conversations:
        print(f"\nUser: {msg}")
        result = model.chat.remote(user_id, msg, img_b64)
        print(f"Level     : {result['level']}")
        print(f"Specialist: {result['specialist']}")
        print(f"Session   : {result['session_count']}/{MAX_CHATS_PER_SESSION}")
        print(f"Has Image : {result.get('has_image')}")
        print(f"Response  :\n{result['response']}")
        print(f"Time      : {result.get('time_ms')}ms")
        if result.get("session_ended"):
            print(">> Session ended.")
        print("-" * 70)
