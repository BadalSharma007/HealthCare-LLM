# ============================================================================
# Qwen Medical AI + LangSmith — app.py
# Model: Qwen/Qwen2-VL-7B-Instruct  (text + image, Hinglish)
# LangSmith: traces every chat, severity, specialist, and model call
# Deploy:  cd "AI Zone Internhsip" && modal deploy Qwen_LangSmith/app.py
# Test:    python3 Qwen_LangSmith/test.py
# ============================================================================
import modal

app = modal.App("qwen-langsmith")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
        "boto3",
        "Pillow",
        "qwen-vl-utils",
        "langsmith",       # LangSmith tracing
    )
    .add_local_file(
        local_path="Qwen_LangSmith/storage_handler.py",
        remote_path="/root/storage_handler.py"
    )
)

sessions = modal.Dict.from_name("qwen-langsmith-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 3
IMAGE_MAX_CHATS       = 5   # extended session only when image + user is unclear

# Words that indicate the user does NOT know what the problem is
UNCLEAR_IMAGE_WORDS = [
    "what is this", "what's this", "what is it", "what's wrong",
    "don't know", "dont know", "not sure", "no idea", "can you tell",
    "identify", "what could", "what could this be", "can you identify",
    "i am confused", "i'm confused", "unclear", "strange", "weird",
    "something wrong", "something on my", "please check", "check this",
    "help me understand", "what do you think", "any idea", "diagnose this",
    "pata nahi", "nahi pata", "kya hai", "kya ho raha", "samajh nahi",
]

def is_unclear_image_query(text: str) -> bool:
    """Returns True if user sent an image but seems unsure about the problem."""
    lower = text.lower()
    return any(phrase in lower for phrase in UNCLEAR_IMAGE_WORDS)

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
    ("sinus",       "ENT Specialist"),
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
# Medicine names to strip from advice
# ============================================================================
MEDICINE_WORDS = [
    "ibuprofen", "paracetamol", "aspirin", "acetaminophen", "naproxen",
    "antibiotic", "steroid", "prescription", "tablet", "pill", "medication",
    "drug", "dose", "mg", "tylenol", "advil", "motrin", "aleve",
]

# ============================================================================
# Hinglish detection — checks Devanagari script or common romanized Hindi words
# ============================================================================
HINDI_WORDS = [
    "mujhe", "mere", "mera", "meri", "kya", "hai", "nahi", "ho", "raha",
    "mein", "aur", "se", "ke", "ki", "ka", "pe", "par", "ko", "hoga",
    "tha", "thi", "kuch", "bahut", "accha", "theek", "dard", "bukhar",
    "pet", "sar", "bimar", "takleef", "abhi", "kal", "aaj", "raat",
    "din", "subah", "zyada", "thoda", "bilkul", "sirf", "bhi", "toh",
    "yeh", "woh", "kaise", "kyun", "kab", "kitna", "kaafi", "bohot",
    "achha", "thik", "nahi", "nahin", "hoon", "hun", "lag", "rahi",
]

def detect_hinglish(text: str) -> bool:
    # Check Devanagari Unicode range
    if any('\u0900' <= c <= '\u097F' for c in text):
        return True
    # Exact word match only — prevents "worse" matching "se", "looks" matching "ko" etc.
    words = set(text.lower().split())
    return bool(words & set(HINDI_WORDS))

LANG_STRICT = (
    "CRITICAL LANGUAGE RULE: You MUST reply ONLY in English or Hinglish (Hindi+English mix). "
    "NEVER use Chinese, Japanese, Korean, Arabic, or any other language. "
    "If the patient writes in any other language, still reply in English only."
)

HINGLISH_INSTRUCTION = (
    "IMPORTANT: The patient is writing in Hinglish (Hindi + English mix). "
    "Respond in Hinglish too. Keep labels like 'Instruction:', 'General Advice:', "
    "'Consult:', 'Keywords:', 'Q1:', 'Q2:' in English, but write all content in Hinglish. "
    "Example: '- Instruction: Aap scratching band karein aur affected area clean rakhein.' "
    + LANG_STRICT
)

# ============================================================================
# Non-symptom words to strip from Keywords line
NON_SYMPTOM_WORDS = [
    "dermatologist", "cardiologist", "neurologist", "orthopedic", "physician",
    "surgeon", "specialist", "doctor", "medical", "evaluation", "treatment",
    "moisturizer", "cream", "lotion", "therapy", "diagnosis", "consultation",
]

def strip_medicines(text: str) -> str:
    # Remove numbered list prefix like "1. " or "2. "
    import re
    text = re.sub(r"^\d+\.\s*", "", text.strip())
    parts = [p.strip() for p in text.replace(";", "|").split("|")]
    # Remove numbered prefix from each part too
    parts = [re.sub(r"^\d+\.\s*", "", p) for p in parts]
    clean = [p for p in parts if p and not any(m in p.lower() for m in MEDICINE_WORDS)]
    return "; ".join(clean) if clean else "Rest and avoid strenuous activity; stay well hydrated"

def clean_keywords(keywords: str) -> str:
    """Remove non-symptom words from keywords line."""
    parts = [k.strip() for k in keywords.split(",")]
    clean = [k for k in parts if k and not any(w in k.lower() for w in NON_SYMPTOM_WORDS)]
    return ", ".join(clean) if clean else keywords

def remove_repetition(text: str) -> str:
    """Remove repeated phrases/sentences within a string."""
    import re as _re
    text = text.strip()
    if not text:
        return text

    # Split on sentence boundaries — handle dot with or without space (Hindi uses both)
    parts = _re.split(r'(?<=[?.!])[\s]+|(?<=\.)\s*(?=[A-ZA-z\u0900-\u097F])', text)
    seen = []
    for part in parts:
        norm = _re.sub(r'\s+', ' ', part.strip().lower())
        if norm and norm not in [_re.sub(r'\s+', ' ', s.lower()) for s in seen]:
            seen.append(part.strip())
    result = " ".join(seen)

    # Catch n-gram repetition: "abc abc abc" → "abc"
    for n in range(len(result) // 2, 8, -1):
        chunk = result[:n]
        if result.replace(chunk, "").strip() == "" or result.startswith(chunk + " " + chunk):
            result = chunk.strip()
            break

    # Catch exact half-string duplication
    half = len(result) // 2
    if len(result) > 20 and result[:half].strip().lower() == result[half:].strip().lower():
        result = result[:half].strip()

    return result

def clean_response_lines(response: str) -> str:
    """Apply remove_repetition to every dash-line in the response."""
    lines = response.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            label, _, content = stripped[2:].partition(":")
            content_clean = remove_repetition(content.strip())
            cleaned.append(f"- {label}: {content_clean}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

# ============================================================================
# Chat 1 prompts — one-shot example format
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
# Image analysis prompt — used when user sends an image
# ============================================================================
IMAGE_ANALYSIS_PROMPT = """You are a medical image analysis assistant.
Write ONE sentence (max 20 words) describing only what is medically visible in this image.
Use clinical terms: redness, swelling, lesion, rash, bruise, wound, inflammation, discolouration.
Do NOT add advice, hashtags, emojis, or social media text. ONE sentence only.
Observation:"""

# ============================================================================
# Follow-up prompts (Chat 2/3) — 3 focused calls, Python assembles format
# ============================================================================
FOLLOWUP_ADVICE_PROMPT = """You are a {specialist} giving lifestyle advice to a patient with {keywords}.
Patient says: {message}

Write EXACTLY 2 short tips separated by a semicolon. No medicines. No numbering. No extra lines.
Format: [tip one]; [tip two]
Example: Apply ice for 15 minutes; keep the area elevated when resting.
{lang_note}
Tips:"""

FOLLOWUP_Q1_PROMPT = """You are a {specialist} asking a follow-up question to a patient.
The patient said: {message}
Their condition involves: {keywords}

Ask ONE short question about how long they have had these symptoms or how severe the pain is.
Do NOT repeat all symptoms in the question. Keep it brief and natural.
{lang_note}
Question:"""

FOLLOWUP_Q2_PROMPT = """You are a {specialist} asking a follow-up question to a patient.
The patient said: {message}
Their condition involves: {keywords}

Ask ONE short, specific question directly related to their exact symptom — about associated symptoms, triggers, or what makes it better/worse.
Examples for stomach pain: "Kya khane ke baad dard badh jaata hai?" or "Do you have nausea or vomiting along with the pain?"
Examples for headache: "Kya roshni ya awaaz se dard badh jaata hai?" or "Is the pain on one side or both sides?"
Do NOT ask generic questions like "koi aur body part affected hai". Make it specific to {keywords}.
Keep it brief and natural.
{lang_note}
Question:"""

# ============================================================================
# Extended image session prompts (Chat 3 & 4 when user sends image)
# ============================================================================
DEEPER_Q1_PROMPT = """You are a {specialist} asking a follow-up question to a patient who shared a medical image.
Image shows: {image_observation}
Patient's condition: {keywords}
Patient's update: {message}

Ask ONE focused question about whether the condition is getting better or worse, or any new symptoms appearing.
Keep it brief and natural.
{lang_note}
Question:"""

DEEPER_Q2_PROMPT = """You are a {specialist} asking a follow-up question to a patient who shared a medical image.
Image shows: {image_observation}
Patient's condition: {keywords}
Patient's update: {message}

Ask ONE question about daily activities or lifestyle factors that may be worsening or causing this condition.
Keep it brief and natural.
{lang_note}
Question:"""

DIAGNOSIS_GUESS_PROMPT = """You are a {specialist} summarizing a patient case after reviewing their image and symptoms.
Image observation: {image_observation}
Symptoms and keywords: {keywords}
All patient information gathered: {message}

Write a brief clinical assessment in this format:
- Possible Condition: [1-2 possible conditions this may suggest — use "may indicate" or "could suggest"]
- Reason: [1 brief reason based on the visible signs and symptoms described]
- General Advice: [2 practical lifestyle tips separated by semicolon — no medicine names]

Keep it clear and simple. Do NOT give a definitive diagnosis. Add a note that this is not a replacement for professional consultation.
{lang_note}
Assessment:"""

PERSONAL_KEYWORDS = ["your name", "who are you", "are you human", "do you feel", "your opinion", "how are you"]

# ============================================================================
# Hardcoded emergency follow-up — never rely on model for LEVEL3
# ============================================================================
EMERGENCY_FOLLOWUP = {
    "Cardiologist":      ("Do not move — sit or lie down quietly and loosen any tight clothing.", "Is there someone with you who can call 112 or 911 right now?", "Are you feeling increasing chest pressure or spreading pain to your jaw or arm?"),
    "Pulmonologist":     ("Sit upright, open windows for fresh air, and do not lie flat.", "Is someone with you who can call emergency services immediately?", "Is the difficulty breathing getting worse every few minutes?"),
    "Neurologist":       ("Do not give the patient anything to eat or drink — keep them still and calm.", "Is someone with you right now to call 112 or 911?", "Are symptoms like numbness or vision loss getting worse rapidly?"),
    "General Physician": ("Stay calm, sit down, and do not attempt to drive yourself to hospital.", "Is there someone nearby who can call emergency services for you?", "Are any of your symptoms rapidly getting worse right now?"),
}

def get_emergency_followup(specialist: str) -> tuple:
    return EMERGENCY_FOLLOWUP.get(specialist, EMERGENCY_FOLLOWUP["General Physician"])


@app.cls(
    image=image,
    gpu="L4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("langsmith-secret"),  # LANGCHAIN_API_KEY + LANGCHAIN_PROJECT
    ],
    max_containers=100,
    # min_containers=1,  # uncomment when demoing — costs $1.10/hr on A10G
)
@modal.concurrent(max_inputs=10)
class QwenMedical:

    @modal.enter()
    def load_model(self):
        import torch, os, sys
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        token = os.environ.get("HF_TOKEN", "")
        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # AutoProcessor handles both text and image inputs for Qwen2-VL
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=token
        )

        sys.path.insert(0, "/root")
        from storage_handler import store_output_to_dynamodb, store_output_log_to_s3
        self.store_dynamodb = store_output_to_dynamodb
        self.store_s3       = store_output_log_to_s3

        # LangSmith client — force project name regardless of env var
        os.environ["LANGCHAIN_PROJECT"] = "Qwen_LangSmith"
        from langsmith import Client
        self.ls_client = Client()
        self.ls_project = "Qwen_LangSmith"

    def _run_model(self, system_prompt: str, user_message: str,
                   max_tokens: int = 250, primer: str = "", pil_image=None) -> str:
        """
        Run Qwen2-VL with optional image.
        Qwen uses ChatML format — same as Apollo, so primer completion works great.
        """
        # Build message content
        if pil_image is not None:
            content = [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": f"{system_prompt}\n\n{user_message}"},
            ]
        else:
            content = [{"type": "text", "text": f"{system_prompt}\n\n{user_message}"}]

        messages = [{"role": "user", "content": content}]

        # Qwen2-VL uses ChatML — apply_chat_template adds <|im_start|>assistant\n at end
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if primer:
            text += primer

        # Process image inputs — import only when image is present
        if pil_image is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to("cuda")
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.6,
            no_repeat_ngram_size=6,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(new_tokens, skip_special_tokens=True)
        result = result.replace("```", "").strip()

        # Stop tokens — prevent extra turns or prompt echoing
        for stop in ["<|im_end|>", "<|im_start|>", "\nUser:", "User:",
                     "\nNow write", "\nPatient complaint", "\n- Q3:", "\nQ3:",
                     "\n\n\n", "Assistant:", "<|endoftext|>"]:
            if stop in result:
                result = result.split(stop)[0].strip()

        # Final repetition cleanup
        result = remove_repetition(result)
        return result

    def _detect_severity(self, prompt: str) -> str:
        classify_prompt = """Classify the severity of this health message. Reply with ONLY one word: LEVEL1, LEVEL2, or LEVEL3.

LEVEL1 = mild, common, manageable at home
LEVEL2 = moderate, needs doctor visit soon
LEVEL3 = ONLY true emergencies: heart attack, stroke, can't breathe, unconscious, heavy bleeding

Examples:
"I have a mild headache" -> LEVEL1
"I feel a bit tired" -> LEVEL1
"I have cold and runny nose" -> LEVEL1
"mujhe thoda bukhar hai" -> LEVEL1
"I have back pain for 2 weeks" -> LEVEL2
"I have recurring fever for 3 days" -> LEVEL2
"mujhe pet mein dard ho raha hai" -> LEVEL2
"mujhe pet mein bohot dard ho raha hai" -> LEVEL2
"bohot dard hai" -> LEVEL2
"kal se dard hai" -> LEVEL2
"the pain is very bad" -> LEVEL2
"nothing is helping" -> LEVEL2
"I have crushing chest pain and left arm pain" -> LEVEL3
"I can't breathe at all" -> LEVEL3
"I am unconscious" -> LEVEL3
"coughing blood heavily" -> LEVEL3
"stroke symptoms" -> LEVEL3

IMPORTANT: "bohot dard", "bahut dard", "severe pain" alone are LEVEL2 NOT LEVEL3.
Only use LEVEL3 for life-threatening emergencies.

Message: "{prompt}"
Answer (LEVEL1 or LEVEL2 or LEVEL3):""".format(prompt=prompt)

        messages = [{"role": "user", "content": [{"type": "text", "text": classify_prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to("cuda")
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
        count   = session["count"]

        # Read prev_data early to know if this is an extended image session (affects max chats)
        prev_data        = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": "", "image_session": False, "image_observation": ""})
        # Only extend to 5 chats if Chat 1 has an image AND user is unclear about the problem
        is_image_session = prev_data.get("image_session", False) or bool(
            image_b64 and count == 0 and is_unclear_image_query(message)
        )
        effective_max    = IMAGE_MAX_CHATS if is_image_session else MAX_CHATS_PER_SESSION

        if count >= effective_max:
            return {
                "response": f"Your session of {effective_max} chats has ended. Please start a new session.",
                "level": "SESSION_ENDED",
                "session_count": count,
                "session_ended": True,
                "specialist": None,
                "follow_up_questions": [],
            }

        # Decode image if provided
        pil_image = None
        image_observation = prev_data.get("image_observation", "")  # carry over from Chat 1
        if image_b64:
            try:
                import re as _re
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                raw_obs = self._run_model(
                    IMAGE_ANALYSIS_PROMPT, message,
                    max_tokens=50, pil_image=pil_image
                )
                # Truncate at first sentence end — prevent Instagram/hashtag runoff
                raw_obs = raw_obs.strip()
                m = _re.search(r'[.!?]', raw_obs)
                if m and m.start() > 8:
                    raw_obs = raw_obs[:m.start() + 1].strip()
                # Cut at any hashtag, emoji cue, or social media marker
                for cut_marker in ['#', 'Please consult', 'consult your', '\n']:
                    if cut_marker in raw_obs:
                        raw_obs = raw_obs[:raw_obs.index(cut_marker)].strip().rstrip('.,;')
                # Hard max 150 chars
                raw_obs = raw_obs[:150].strip()
                image_observation = remove_repetition(raw_obs) if raw_obs else ""
                if image_observation:
                    message = f"{message} [Image: {image_observation}]"
            except Exception:
                pil_image = None

        start = time.time()
        severity = self._detect_severity(message)

        prev_keywords   = prev_data.get("keywords", "")
        prev_specialist = prev_data.get("specialist", "")

        # Detect language — use original message before image text was appended
        is_hinglish = detect_hinglish(message)
        lang_note = f"\n\n{HINGLISH_INSTRUCTION}" if is_hinglish else f"\n\n{LANG_STRICT}"

        # Chat 1: detect fresh. Chat 2/3: always reuse Chat 1 specialist.
        if count == 0:
            specialist = detect_specialist(message)
        else:
            specialist = prev_specialist or detect_specialist(prev_keywords) or detect_specialist(message)

        # ----------------------------------------------------------------
        # CHAT 1 — one-shot example prompt → model outputs full dash format
        # ----------------------------------------------------------------
        if count == 0:
            if severity == "LEVEL1":
                system = LEVEL_1_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note
            elif severity == "LEVEL2":
                system = LEVEL_2_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note
            else:
                system = LEVEL_3_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note

            # Pass pil_image=None — image info is already in message as [Image: obs]
            # Passing the image to the long prompt causes the model to generate garbage
            raw = self._run_model(system, message, max_tokens=300, pil_image=None)

            # If model echoed system prompt, extract only dash-formatted block
            if not raw.strip().startswith("- "):
                lines = raw.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("- "):
                        raw = "\n".join(lines[i:])
                        break

            # Hard-truncate after Q2 — discard any garbage after
            truncated = []
            for line in raw.split("\n"):
                truncated.append(line)
                if line.strip().lstrip("- ").upper().startswith("Q2:"):
                    break
            raw = "\n".join(truncated)

            # Ensure all known lines have dash prefix + strip medicines
            DASH_PREFIXES = ["Mild Concern", "Moderate Concern", "EMERGENCY",
                             "Instruction:", "General Advice:", "Consult:",
                             "Keywords:", "Q1:", "Q2:"]
            cleaned_lines = []
            for line in raw.split("\n"):
                s = line.strip()
                if not s:
                    continue
                if not s.startswith("- "):
                    for pfx in DASH_PREFIXES:
                        if s.upper().startswith(pfx.upper()):
                            s = "- " + s
                            break
                s2 = s.lstrip("- ")
                if s2.startswith("General Advice:"):
                    adv = s2.replace("General Advice:", "").strip()
                    cleaned_lines.append(f"- General Advice: {strip_medicines(adv)}")
                else:
                    cleaned_lines.append(s)
            response = "\n".join(cleaned_lines)

            if image_observation:
                response = f"- Image Observation: {image_observation}\n" + response

            # Extract keywords now so Q1/Q2 can use them
            kw_for_q = ""
            for line in response.split("\n"):
                s = line.strip().lstrip("- ")
                if s.startswith("Keywords:"):
                    kw_for_q = s.replace("Keywords:", "").strip()
                    break
            if not kw_for_q:
                kw_for_q = message

            # Always generate Q1/Q2 separately — model often stops before them
            def fmt(prompt, **kwargs):
                for k, v in kwargs.items():
                    prompt = prompt.replace("{" + k + "}", v)
                return prompt

            try:
                q1_raw = self._run_model(
                    fmt(FOLLOWUP_Q1_PROMPT, specialist=specialist, keywords=kw_for_q, message=message, lang_note=lang_note),
                    message, max_tokens=60, primer="Question:"
                )
                q1_text = remove_repetition(q1_raw.split("\n")[0].strip())
            except Exception:
                q1_text = ""
            if not q1_text:
                q1_text = "Yeh symptoms kitne time se hain aur kitna severe hai?" if is_hinglish else "Can you describe when the symptoms started and how severe they are?"

            try:
                q2_raw = self._run_model(
                    fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw_for_q, message=message, lang_note=lang_note),
                    message, max_tokens=60, primer="Question:"
                )
                q2_text = remove_repetition(q2_raw.split("\n")[0].strip())
            except Exception:
                q2_text = ""
            if not q2_text:
                # Retry once with just keywords as context
                try:
                    q2_retry = self._run_model(
                        fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw_for_q, message=kw_for_q, lang_note=lang_note),
                        kw_for_q, max_tokens=60, primer="Question:"
                    )
                    q2_text = remove_repetition(q2_retry.split("\n")[0].strip())
                except Exception:
                    q2_text = ""
            if not q2_text:
                q2_text = (f"Kya {kw_for_q} ke saath koi aur symptoms bhi hain?" if is_hinglish
                           else f"Are there any other symptoms along with {kw_for_q}?")

            # Strip any Q1/Q2 already in response to avoid duplicates, then append fresh ones
            filtered = [l for l in response.split("\n")
                        if not l.strip().lstrip("- ").upper().startswith("Q1:")
                        and not l.strip().lstrip("- ").upper().startswith("Q2:")]
            response = "\n".join(filtered).rstrip() + f"\n- Q1: {q1_text}\n- Q2: {q2_text}"

        # ----------------------------------------------------------------
        # CHAT 2/3 — completion-style, Python assembles dash format
        # ----------------------------------------------------------------
        else:
            kw = prev_keywords or "your symptoms"

            if severity == "LEVEL3":
                advice, q1, q2 = get_emergency_followup(specialist)
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
                    fmt(FOLLOWUP_ADVICE_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    message, max_tokens=80, primer="Tips:", pil_image=pil_image
                )
                advice_text = strip_medicines(advice_raw.split("\n")[0].strip())
                if not advice_text:
                    advice_text = "Aaram karein aur paani zyada piyein" if is_hinglish else "Rest and avoid strenuous activity; stay well hydrated"

                q1_raw = self._run_model(
                    fmt(FOLLOWUP_Q1_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    message, max_tokens=60, primer="Question:"
                )
                q1_text = remove_repetition(q1_raw.split("\n")[0].strip())
                if not q1_text:
                    q1_text = "Yeh problem kitne time se hai aur severity kya hai?" if is_hinglish else "Can you describe how your symptom has changed since it started?"

                q2_raw = self._run_model(
                    fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    message, max_tokens=60, primer="Question:"
                )
                q2_text = remove_repetition(q2_raw.split("\n")[0].strip())
                if not q2_text:
                    q2_text = "Aur koi body part bhi affected hai kya?" if is_hinglish else "Have you noticed any other symptoms appearing alongside this?"

                is_last_chat   = (count == effective_max - 1)
                is_deeper_chat = is_image_session and count == 2   # Chat 3 — deeper symptom questions
                is_diag_chat   = is_image_session and count == 3   # Chat 4 — diagnosis guess

                if is_diag_chat:
                    # ── Chat 4 (image session) — diagnosis guess ──────────────
                    diag_raw = self._run_model(
                        fmt(DIAGNOSIS_GUESS_PROMPT,
                            specialist=specialist, keywords=kw,
                            message=message, lang_note=lang_note,
                            image_observation=image_observation or "Not provided"),
                        message, max_tokens=200, primer="Assessment:"
                    )
                    # Clean into dash format
                    diag_lines = []
                    for line in diag_raw.split("\n"):
                        s = line.strip()
                        if s and not s.startswith("- "):
                            s = "- " + s
                        if s:
                            diag_lines.append(s)
                    diag_text = "\n".join(diag_lines) if diag_lines else f"- Possible Condition: This may indicate a {kw}-related condition — please consult a {specialist} for confirmation."
                    response = (
                        f"- {kw}: {specialist}\n"
                        f"{diag_text}\n"
                        f"- Note: This is not a diagnosis. Please consult a {specialist} for professional evaluation."
                    )
                elif is_last_chat:
                    # ── Final chat — booking recommendation ───────────────────
                    response = (
                        f"- {kw}: {specialist} + {advice_text}\n"
                        f"- Next Step: Book an appointment with a {specialist} as soon as possible.\n"
                        f"- Book via: Practo / Apollo247 / nearest hospital\n"
                        f"- Emergency: Call 112 (India) or 108 (Ambulance) if symptoms worsen"
                    )
                elif is_deeper_chat:
                    # ── Chat 3 (image session) — deeper symptom Q1/Q2 ─────────
                    q1_raw  = self._run_model(
                        fmt(DEEPER_Q1_PROMPT, specialist=specialist, keywords=kw,
                            message=message, lang_note=lang_note,
                            image_observation=image_observation or "Not provided"),
                        message, max_tokens=60, primer="Question:"
                    )
                    q1_text = q1_raw.split("\n")[0].strip()
                    if not q1_text:
                        q1_text = "Is the condition getting worse or staying the same since it started?"

                    q2_raw  = self._run_model(
                        fmt(DEEPER_Q2_PROMPT, specialist=specialist, keywords=kw,
                            message=message, lang_note=lang_note,
                            image_observation=image_observation or "Not provided"),
                        message, max_tokens=60, primer="Question:"
                    )
                    q2_text = q2_raw.split("\n")[0].strip()
                    if not q2_text:
                        q2_text = "Are there any activities or positions that make it worse or better?"

                    response = (
                        f"- {kw}: {specialist} + {advice_text}\n"
                        f"- Q1: {q1_text}\n"
                        f"- Q2: {q2_text}"
                    )
                else:
                    # ── Regular Chat 2 follow-up ──────────────────────────────
                    response = (
                        f"- {kw}: {specialist} + {advice_text}\n"
                        f"- Q1: {q1_text}\n"
                        f"- Q2: {q2_text}"
                    )

                if image_observation and not is_diag_chat:
                    response = f"- Image Observation: {image_observation}\n" + response

        # Clean repetition from every line of the response
        response = clean_response_lines(response)

        # Extract keywords and follow-up questions
        keywords = ""
        follow_up_questions = []
        for line in response.split("\n"):
            s = line.strip().lstrip("- ")
            if s.startswith("Keywords:"):
                keywords = clean_keywords(s.replace("Keywords:", "").strip())
            elif s.upper().startswith("Q1:"):
                follow_up_questions.append(s[3:].strip())
            elif s.upper().startswith("Q2:"):
                follow_up_questions.append(s[3:].strip())

        if not keywords:
            keywords = prev_keywords

        # Emergency banner for Chat 1 LEVEL3
        if severity == "LEVEL3" and count == 0:
            response += (
                f"\n\n🚨 EMERGENCY — CALL NOW:\n"
                f"- Ambulance (India): 108\n"
                f"- National Emergency: 112\n"
                f"- Go to nearest hospital emergency — ask for {specialist}\n"
                f"- Book appointment: Apollo247 / Practo / nearest hospital"
            )

        elapsed_ms = round((time.time() - start) * 1000)
        new_count  = count + 1
        sessions[user_id] = {"count": new_count}
        session_ended = new_count >= effective_max

        # ----------------------------------------------------------------
        # LangSmith — log this chat as a traced run
        # ----------------------------------------------------------------
        try:
            import uuid
            run_id = str(uuid.uuid4())
            self.ls_client.create_run(
                id=run_id,
                name="medical_chat",
                run_type="chain",
                project_name=self.ls_project,
                inputs={
                    "user_id":    user_id,
                    "message":    message,
                    "chat_num":   new_count,
                    "has_image":  bool(image_b64),
                    "is_hinglish": is_hinglish,
                },
                outputs={
                    "response":   response,
                    "level":      severity,
                    "specialist": specialist,
                    "keywords":   keywords,
                    "follow_up_questions": follow_up_questions,
                    "time_ms":    elapsed_ms,
                },
                tags=[severity, specialist, f"chat_{new_count}",
                      "hinglish" if is_hinglish else "english"],
                extra={"metadata": {
                    "model":       "Qwen2-VL-7B-Instruct",
                    "session_end": session_ended,
                }},
            )
            self.ls_client.update_run(run_id, end_time=__import__("datetime").datetime.utcnow())
        except Exception:
            pass  # never let LangSmith errors break the chat

        chat_id      = f"chat_{user_id}"
        session_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": "", "image_session": False, "image_observation": ""})
        session_data["keywords"]        = keywords
        session_data["specialist"]      = specialist
        session_data["image_session"]   = is_image_session
        session_data["image_observation"] = image_observation
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
            "model":              "Qwen2-VL-7B-Instruct-LangSmith",
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
    model = QwenMedical()
    user_id = "qwen_test_001"
    model.reset_session.remote(user_id)

    conversations = [
        ("I have a mild headache since morning", ""),
        ("I have severe back pain for 3 weeks, cant walk properly", ""),
        ("I am having crushing chest pain and my left arm is numb", ""),
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
