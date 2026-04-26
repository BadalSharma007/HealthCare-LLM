# ============================================================================
# Apollo Medical AI — app.py
# Model: FreedomIntelligence/Apollo-7B
# Deploy:  cd "AI Zone Internhsip" && modal deploy Apollo/app.py
# Test:    python3 Apollo/test.py
# ============================================================================
import modal

app = modal.App("apollo-medical")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
        "boto3",
        "langsmith",
    )
    .add_local_file(
        local_path="Apollo/storage_handler.py",
        remote_path="/root/storage_handler.py"
    )
)

sessions = modal.Dict.from_name("apollo-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 3

# ============================================================================
# Expanded specialist map — sorted longest phrase first so multi-word phrases
# match before single words (e.g. "chest pain" before "chest").
# ============================================================================
SPECIALIST_MAP = [
    # ── Specific conditions first (before generic single words) ──────────────
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
LANG_STRICT = (
    "CRITICAL LANGUAGE RULE: You MUST reply ONLY in English or Hinglish (Hindi+English mix). "
    "NEVER use Chinese, Japanese, Korean, Arabic, or any other language. "
    "If the patient writes in any other language, still reply in English only."
)

FORMAT_RULE = """
STRICT FORMAT RULES — follow exactly:
- Output ONLY dash-prefixed lines. No free text, no paragraphs, no hashtags, no emojis.
- Each line must start with "- Label: value"
- STOP immediately after Q2. Never write Q3 or anything after Q2.
- Keep each line SHORT — max 2 sentences.
- Do NOT repeat yourself. Do NOT add motivational text or sign-offs.
""" + LANG_STRICT

# ============================================================================
# Chat 1 — focused completion prompts (one per field).
# Python assembles the full dash format. Model only generates one short value.
# ============================================================================
C1_INSTRUCTION_PROMPT = (
    "Patient complaint: {complaint}\n"
    "You are a {specialist}. Give ONE clear action the patient should take right now. Max 20 words. No medicine names.\n"
    + LANG_STRICT + "\nAction:"
)
C1_ADVICE_PROMPT = (
    "Patient complaint: {complaint}\n"
    "You are a {specialist}. Give 2 short home-care tips. Separate with semicolon. Max 15 words each. No medicine names.\n"
    + LANG_STRICT + "\nTips:"
)
C1_KEYWORDS_PROMPT = (
    "Extract 3-5 medical keywords from this complaint (comma-separated, no explanation): {complaint}\n"
    "Keywords:"
)
C1_Q1_PROMPT = (
    "Patient complaint: {complaint}\n"
    "You are a {specialist}. Ask ONE specific question about duration or timing of the symptom. Max 15 words.\n"
    + LANG_STRICT + "\nQuestion:"
)
C1_Q2_PROMPT = (
    "Patient complaint: {complaint}. Keywords: {keywords}\n"
    "You are a {specialist}. Ask ONE question about any other related symptoms the patient may have. Max 15 words.\n"
    + LANG_STRICT + "\nQuestion:"
)

SEVERITY_LABEL = {"LEVEL1": "Mild Concern", "LEVEL2": "Moderate Concern", "LEVEL3": "EMERGENCY"}

# ============================================================================
# Follow-up prompt (Chat 2/3) — completion style, Python assembles dash format.
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
    "Cardiologist":       ("Do not move — sit or lie down quietly and loosen any tight clothing.", "Is there someone with you who can call 112 or 911 right now?", "Are you feeling increasing chest pressure or spreading pain to your jaw or arm?"),
    "Pulmonologist":      ("Sit upright, open windows for fresh air, and do not lie flat.", "Is someone with you who can call emergency services immediately?", "Is the difficulty breathing getting worse every few minutes?"),
    "Neurologist":        ("Do not give the patient anything to eat or drink — keep them still and calm.", "Is someone with you right now to call 112 or 911?", "Are symptoms like numbness or vision loss getting worse rapidly?"),
    "General Physician":  ("Stay calm, sit down, and do not attempt to drive yourself to hospital.", "Is there someone nearby who can call emergency services for you?", "Are any of your symptoms rapidly getting worse right now?"),
}

def get_emergency_followup(specialist: str, keywords: str) -> tuple:
    advice, q1, q2 = EMERGENCY_FOLLOWUP.get(specialist, EMERGENCY_FOLLOWUP["General Physician"])
    return advice, q1, q2


@app.cls(
    image=image,
    gpu="L4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("langsmith-secret"),
    ],
    max_containers=100,
    # min_containers=1,  # uncomment when demoing — costs $1.10/hr on A10G
)
@modal.concurrent(max_inputs=10)
class ApolloModel:

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

        # LangSmith client — force project name to folder name
        os.environ["LANGCHAIN_PROJECT"] = "Apollo"
        from langsmith import Client
        self.ls_client  = Client()
        self.ls_project = "Apollo"

    def _run_model(self, system_prompt: str, user_message: str, max_tokens: int = 250, primer: str = "") -> str:
        # primer = text to prepend to assistant response (forces completion style)
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n{primer}"
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=6,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = result.replace("```", "").strip()
        for stop in ["\nUser:", "\n<|im_start|>", "<|im_end|>", "User:",
                     "\nNow write", "\nPatient complaint", "\n- Q3:", "\nQ3:",
                     "\n\n\n", "Assistant:", "Note:", "Disclaimer:"]:
            if stop in result:
                result = result.split(stop)[0].strip()
        return result

    def _complete(self, prompt: str, max_tokens: int = 80) -> str:
        """Raw text completion — no chat template. Used for focused single-field prompts."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for stop in ["\n\n", "Patient:", "User:", "Question:", "Tips:", "Action:", "Keywords:", "<|"]:
            if stop in result:
                result = result.split(stop)[0].strip()
        return result.split("\n")[0].strip()

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

        inputs = self.tokenizer(classify_prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).upper().strip()
        if "LEVEL3" in raw:
            return "LEVEL3"
        elif "LEVEL2" in raw:
            return "LEVEL2"
        return "LEVEL1"

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

        start = time.time()
        severity = self._detect_severity(message)

        prev_data    = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": ""})
        prev_keywords   = prev_data.get("keywords", "")
        prev_specialist = prev_data.get("specialist", "")

        # Chat 1: detect from message. Chat 2/3: always reuse Chat 1's specialist.
        if count == 0:
            specialist = detect_specialist(message)
        else:
            # Use saved specialist from Chat 1; re-detect only if not saved
            specialist = prev_specialist or detect_specialist(prev_keywords) or detect_specialist(message)

        # ----------------------------------------------------------------
        # CHAT 1 — one focused call per field, Python assembles dash format
        # ----------------------------------------------------------------
        if count == 0:
            def c1(template, **kwargs):
                prompt = template
                for k, v in kwargs.items():
                    prompt = prompt.replace("{" + k + "}", v)
                return self._complete(prompt, max_tokens=80)

            label = SEVERITY_LABEL.get(severity, "Moderate Concern")

            instruction = c1(C1_INSTRUCTION_PROMPT, complaint=message, specialist=specialist)
            if not instruction:
                instruction = "Rest at home and monitor your symptoms closely."

            advice_raw = c1(C1_ADVICE_PROMPT, complaint=message, specialist=specialist)
            advice = strip_medicines(advice_raw) if advice_raw else "Stay hydrated; avoid strenuous activity"

            keywords_raw = c1(C1_KEYWORDS_PROMPT, complaint=message)
            keywords = keywords_raw if keywords_raw else message[:50]

            q1 = c1(C1_Q1_PROMPT, complaint=message, specialist=specialist)
            if not q1:
                q1 = "How long have you been experiencing this symptom?"

            q2 = c1(C1_Q2_PROMPT, complaint=message, keywords=keywords, specialist=specialist)
            if not q2:
                q2 = f"Are there any other symptoms along with {keywords.split(',')[0].strip()}?"

            response = (
                f"- {label}\n"
                f"- Instruction: {instruction}\n"
                f"- General Advice: {advice}\n"
                f"- Consult: [{specialist}]\n"
                f"- Keywords: {keywords}\n"
                f"- Q1: {q1}\n"
                f"- Q2: {q2}"
            )

        # ----------------------------------------------------------------
        # CHAT 2/3 — completion-style prompt, Python assembles dash format
        # ----------------------------------------------------------------
        else:
            kw = prev_keywords or "your symptoms"

            if severity == "LEVEL3":
                # Hardcoded emergency follow-up — never rely on model for emergencies
                advice, q1, q2 = get_emergency_followup(specialist, kw)
                response = (
                    f"- {kw}: {specialist}\n"
                    f"- 🚨 EMERGENCY: Call 112 (India) or 911 (US) NOW or go to the nearest hospital.\n"
                    f"- Immediate Action: {advice}\n"
                    f"- Q1: {q1}\n"
                    f"- Q2: {q2}"
                )
            else:
                def fmt(prompt, **kwargs):
                    for k, v in kwargs.items():
                        prompt = prompt.replace("{" + k + "}", v)
                    return prompt

                # 3 separate focused prompts → each gets one clean answer
                advice_raw = self._run_model(
                    fmt(FOLLOWUP_ADVICE_PROMPT, specialist=specialist, keywords=kw, message=message),
                    message, max_tokens=80, primer="Tips:"
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

        # Carry over keywords for follow-up chats
        if not keywords:
            keywords = prev_keywords

        # Emergency banner for Chat 1 LEVEL3
        if severity == "LEVEL3" and count == 0:
            response += (
                f"\n\n🚨 EMERGENCY: Please call 112 (India) / 911 (US) immediately "
                f"or go to the nearest hospital. Ask for a {specialist}."
            )

        elapsed_ms = round((time.time() - start) * 1000)
        new_count    = count + 1
        sessions[user_id] = {"count": new_count}
        session_ended = new_count >= MAX_CHATS_PER_SESSION

        # LangSmith — log this chat as a traced run
        try:
            import uuid
            run_id = str(uuid.uuid4())
            self.ls_client.create_run(
                id=run_id,
                name="medical_chat",
                run_type="chain",
                project_name=self.ls_project,
                inputs={
                    "user_id":   user_id,
                    "message":   message,
                    "chat_num":  new_count,
                },
                outputs={
                    "response":            response,
                    "level":               severity,
                    "specialist":          specialist,
                    "keywords":            keywords,
                    "follow_up_questions": follow_up_questions,
                    "time_ms":             elapsed_ms,
                },
                tags=[severity, specialist, f"chat_{new_count}"],
                extra={"metadata": {"model": "Apollo-7B", "session_end": session_ended}},
            )
            self.ls_client.update_run(run_id, end_time=__import__("datetime").datetime.utcnow())
        except Exception:
            pass  # never let LangSmith errors break the chat

        # Save session record
        chat_id      = f"chat_{user_id}"
        session_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": ""})
        session_data["keywords"]   = keywords
        session_data["specialist"] = specialist
        session_data["chats"][f"query_{new_count}"] = {
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
            "response":            response,
            "level":               severity,
            "session_count":       new_count,
            "session_ended":       session_ended,
            "specialist":          specialist,
            "follow_up_questions": follow_up_questions,
            "time_ms":             elapsed_ms,
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
        return self.chat.local(user_id, message)


@app.local_entrypoint()
def main():
    model = ApolloModel()
    user_id = "apollo_test_001"
    model.reset_session.remote(user_id)

    conversations = [
        "I have a mild headache since morning",
        "I've had severe back pain for 3 weeks, can't walk properly",
        "I'm having crushing chest pain and my left arm is numb",
    ]

    for msg in conversations:
        print(f"\nUser: {msg}")
        result = model.chat.remote(user_id, msg)
        print(f"Level     : {result['level']}")
        print(f"Specialist: {result['specialist']}")
        print(f"Session   : {result['session_count']}/{MAX_CHATS_PER_SESSION}")
        print(f"Response  :\n{result['response']}")
        print(f"Time      : {result.get('time_ms')}ms")
        if result.get("session_ended"):
            print(">> Session ended.")
        print("-" * 70)
