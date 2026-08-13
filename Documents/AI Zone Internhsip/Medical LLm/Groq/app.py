# ============================================================================
# Groq Medical AI — app.py
# Model: qwen/qwen3.6-27b on Groq (text + image, Hinglish)
# Same business logic/prompts as Qwen/app.py — only the inference backend
# differs: instead of loading Qwen2-VL-7B locally on a GPU, this calls
# Groq's hosted API (fast LPU inference, no cold start, no GPU needed).
# Deploy:  cd "AI Zone Internhsip" && modal deploy Groq/app.py
# ============================================================================
import modal

app = modal.App("groq-medical")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests",
        "fastapi[standard]",
        "faiss-cpu",             # vector similarity search (free, no server)
        "sentence-transformers", # embedding model (free, runs on CPU)
        "numpy",                 # required by faiss + sentence-transformers
    )
)

sessions = modal.Dict.from_name("groq-medical-sessions", create_if_missing=True)
MAX_CHATS_PER_SESSION = 4

GROQ_MODEL = "qwen/qwen3.6-27b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============================================================================
# Expanded specialist map — ordered list so multi-word phrases match before
# single words (e.g. "chest pain" before "chest", "pilonidal sinus" before "sinus").
# (Identical to Qwen/app.py — kept in sync for structural parity.)
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
    # ── Hinglish body-part terms ─────────────────────────────────────────────
    ("sar dard",  "Neurologist"),
    ("sir dard",  "Neurologist"),
    ("pet dard",  "Gastroenterologist"),
    ("aankhon",   "Ophthalmologist"),
    ("aankh",     "Ophthalmologist"),
    ("ankh",      "Ophthalmologist"),
    ("kaan",      "ENT Specialist"),
    ("naak",      "ENT Specialist"),
    ("gala",      "ENT Specialist"),
    ("gale",      "ENT Specialist"),
    ("daant",     "Dentist"),
    ("dant",      "Dentist"),
    ("kamar",     "Orthopedic Surgeon"),
    ("gardan",    "Orthopedic Surgeon"),
    ("ghutna",    "Orthopedic Surgeon"),
    ("ghutne",    "Orthopedic Surgeon"),
    ("seene",     "Cardiologist"),
    ("seena",     "Cardiologist"),
    ("chhati",    "Cardiologist"),
    ("saans",     "Pulmonologist"),
    ("khansi",    "Pulmonologist"),
    ("khujli",    "Dermatologist"),
    ("peshab",    "Urologist"),
    ("ulti",      "Gastroenterologist"),
    ("dast",      "Gastroenterologist"),
    ("pet",       "Gastroenterologist"),
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

def detect_specialist_strict(text: str):
    """Whole-word keyword match (so 'ear' never matches inside 'appears').
    Returns None when no keyword matches."""
    import re
    lower = text.lower()
    for keyword, specialist in SPECIALIST_MAP:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return specialist
    return None


def detect_specialist(text: str) -> str:
    return detect_specialist_strict(text) or "General Physician"

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
    if any('ऀ' <= c <= 'ॿ' for c in text):
        return True
    words = set(text.lower().split())
    return bool(words & set(HINDI_WORDS))

HINGLISH_INSTRUCTION = (
    "IMPORTANT: The patient is writing in Hinglish (Hindi + English mix). "
    "Respond in Hinglish too. Keep labels like 'Instruction:', 'General Advice:', "
    "'Consult:', 'Keywords:', 'Q1:', 'Q2:' in English, but write all content in Hinglish. "
    "Example: '- Instruction: Aap scratching band karein aur affected area clean rakhein.'"
)

NON_SYMPTOM_WORDS = [
    "dermatologist", "cardiologist", "neurologist", "orthopedic", "physician",
    "surgeon", "specialist", "doctor", "medical", "evaluation", "treatment",
    "moisturizer", "cream", "lotion", "therapy", "diagnosis", "consultation",
]

def strip_medicines(text: str) -> str:
    import re
    text = re.sub(r"^\d+\.\s*", "", text.strip())
    parts = [p.strip() for p in text.replace(";", "|").split("|")]
    parts = [re.sub(r"^\d+\.\s*", "", p) for p in parts]
    clean = [p for p in parts if p and not any(m in p.lower() for m in MEDICINE_WORDS)]
    return "; ".join(clean) if clean else "Rest and avoid strenuous activity; stay well hydrated"

def clean_keywords(keywords: str) -> str:
    parts = [k.strip() for k in keywords.split(",")]
    clean = [k for k in parts if k and not any(w in k.lower() for w in NON_SYMPTOM_WORDS)]
    return ", ".join(clean) if clean else keywords

# ============================================================================
# Chat 1 prompts — one-shot example format (identical to Qwen/app.py)
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

IMAGE_ANALYSIS_PROMPT = """You are a medical image analysis assistant. The patient has shared a medical image.
Describe what you observe in the image in simple terms a patient can understand.
Focus on: visible symptoms, affected area, any visible abnormality.
Keep it brief — 2-3 sentences. Do NOT suggest diagnosis or medicines.
Image observation:"""

FOLLOWUP_ADVICE_PROMPT = """You are a {specialist} giving lifestyle advice to a patient with {keywords}.
Patient says: {message}

Write exactly 2 short practical tips to help the patient manage {keywords} at home.
Rules: No medicine names. No numbering. Separate the 2 tips with a semicolon only.
Example format: Keep the area clean and dry; avoid scratching and use a cold compress for relief.
{lang_note}
Reply with only the two tips, nothing else.
Tips:"""

FOLLOWUP_Q1_PROMPT = """You are a {specialist} asking a follow-up question to a patient with {keywords}.
The patient said: {message}

Ask ONE specific question about how long they have had {keywords} or how severe it is.
{lang_note}
Reply with only the question, nothing else.
Question:"""

FOLLOWUP_Q2_PROMPT = """You are a {specialist} asking about other body areas to a patient with {keywords}.
The patient said: {message}

Ask ONE question about whether any other part of their body is also affected or showing symptoms.
{lang_note}
Reply with only the question, nothing else.
Question:"""

PERSONAL_KEYWORDS = ["your name", "who are you", "are you human", "do you feel", "your opinion", "how are you"]

# ============================================================================
# Conversation continuity — decide (semantically, not by keyword) whether a
# follow-up message CONTINUES the current complaint or starts a NEW one.
# ============================================================================
CONTINUATION_PROMPT = """A patient is in an ongoing medical consultation.

Previous complaint: "{prev}"
Previous keywords: {keywords}
New message from patient: "{message}"

Decide if the new message is:
- CONTINUE = it adds detail to the SAME complaint (duration, location, severity, colour, itching, pain score, swelling, fever duration), or answers a follow-up question.
- NEW = it clearly introduces a DIFFERENT symptom or body problem.

Rules:
- Words like "ye", "yeh", "isko", "waha", "wahan", "ab", "still", "same", "it", "this" refer to the PREVIOUS complaint -> CONTINUE.
- A body-part location added to an existing complaint (e.g. "ye gardan pe hai" = the rash is on the neck) is a detail -> CONTINUE.
- Only answer NEW when a clearly different symptom is introduced.

Answer with ONLY one word: CONTINUE or NEW."""

# ============================================================================
# Safety: red-flag phrases that FORCE an emergency classification, and
# self-harm phrases that route to a crisis-support response. Keyword-based on
# purpose — these must never depend on a model call that could misfire.
# ============================================================================
EMERGENCY_OVERRIDE = [
    "chest pain", "left arm", "can't breathe", "cannot breathe", "not breathing",
    "unconscious", "unresponsive", "heart attack", "stroke", "severe bleeding",
    "coughing blood", "vomiting blood", "blue lips", "face drooping", "slurred speech",
    "seene me dard", "saans nahi", "behosh", "khoon ki ulti", "bleeding ruk nahi",
]

SELF_HARM_WORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self harm",
    "self-harm", "hurt myself", "no reason to live",
    "khudkushi", "aatmhatya", "marna chahta", "jeena nahi chahta", "jaan de",
]

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way, and I'm glad you reached out. "
    "You are not alone and help is available right now.\n"
    "• India — KIRAN helpline: 1800-599-0019 (24/7, free, confidential)\n"
    "• AASRA: +91-9820466726\n"
    "• If you are in immediate danger, please call 112 now.\n"
    "Talking to someone you trust or a trained counsellor can really help — please reach out to one of the numbers above."
)

# ============================================================================
# Greeting detection — plain "hi"/"hello" etc. get a natural chat reply and
# do NOT consume a session slot. The 4-chat session only starts once the
# patient actually describes a health concern.
# ============================================================================
GREETING_MESSAGES = {
    "hi", "hii", "hiii", "hiiii", "hello", "helo", "hey", "heyy", "heya",
    "hi there", "hii there", "hello there", "hey there",
    "yo", "sup", "whats up", "what's up",
    "good morning", "good afternoon", "good evening", "gm", "ge",
    "namaste", "hola", "greetings", "howdy",
}

def is_greeting(text: str) -> bool:
    import re
    normalized = re.sub(r"[^\w\s']", "", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in GREETING_MESSAGES

GREETING_REPLY = "Hi! 👋 How can I help you with your health today? Feel free to describe any symptoms or health concerns you have."

# ============================================================================
# RAG — Medical Knowledge Base + FAISS index
# A simple retrieval-augmented generation pipeline:
#   1. Embed user query with sentence-transformers (all-MiniLM-L6-v2)
#   2. Cosine-search FAISS index over MEDICAL_KB chunks
#   3. Inject top-3 retrieved facts into the LLM prompt as grounding context
# All free: no external vector DB, no API key, runs on CPU inside Modal.
# ============================================================================
MEDICAL_KB = [
    # ── General / Fever ──────────────────────────────────────────────────────
    "Fever above 103°F (39.4°C) lasting more than 3 days in adults requires medical evaluation.",
    "Viral fevers typically resolve in 3–5 days; bacterial fevers often require antibiotic treatment.",
    "Paracetamol (acetaminophen) is the first-line agent for fever management in adults and children.",
    "Dehydration during fever can worsen symptoms; drinking ORS or water frequently is essential.",
    "Febrile seizures can occur in children aged 6 months to 5 years when temperature rises rapidly.",
    # ── Headache / Neurological ───────────────────────────────────────────────
    "Migraines are recurrent headaches often accompanied by nausea, vomiting, and light sensitivity.",
    "Tension headaches cause a dull, pressing pain around the forehead and are the most common headache type.",
    "A sudden, severe 'thunderclap' headache — the worst of one's life — is a red flag for subarachnoid hemorrhage.",
    "Cluster headaches cause intense one-sided pain around the eye and typically occur in cycles.",
    "Chronic daily headaches occurring more than 15 days per month may indicate medication overuse headache.",
    # ── Chest / Cardiac ───────────────────────────────────────────────────────
    "Classic heart attack symptoms include crushing chest pain, left arm pain, jaw pain, and sweating.",
    "Atypical heart attack symptoms in women include fatigue, shortness of breath, and nausea without chest pain.",
    "Angina pectoris is chest pain caused by reduced blood flow to the heart, often triggered by exertion.",
    "Hypertension (high blood pressure) is a major risk factor for heart attack, stroke, and kidney disease.",
    "Palpitations — a rapid or irregular heartbeat — can be caused by anxiety, caffeine, or arrhythmias.",
    # ── Respiratory ───────────────────────────────────────────────────────────
    "Asthma is a chronic inflammatory airway disease causing wheezing, breathlessness, and cough.",
    "Pneumonia presents with fever, productive cough, chest pain, and reduced oxygen saturation.",
    "COPD is a progressive lung disease caused mainly by smoking; symptoms include chronic cough and dyspnea.",
    "Shortness of breath at rest, cyanosis (blue lips), or rapid breathing are emergency warning signs.",
    "Pulmonary embolism (blood clot in lung) causes sudden shortness of breath, chest pain, and rapid heart rate.",
    # ── Abdomen / Gastro ──────────────────────────────────────────────────────
    "Gastroesophageal reflux disease (GERD) causes heartburn, regurgitation, and sometimes chronic cough.",
    "Irritable bowel syndrome (IBS) causes abdominal pain, bloating, diarrhea, or constipation without structural damage.",
    "Appendicitis typically starts with pain around the navel that migrates to the lower right abdomen.",
    "Peptic ulcer disease causes burning epigastric pain that may improve or worsen with eating.",
    "Acute pancreatitis presents with severe upper abdominal pain radiating to the back, nausea, and vomiting.",
    # ── Skin / Dermatology ────────────────────────────────────────────────────
    "Eczema (atopic dermatitis) is a chronic inflammatory skin condition causing dry, itchy, and inflamed skin.",
    "Psoriasis causes scaly, silvery plaques commonly on elbows, knees, and scalp due to rapid skin cell turnover.",
    "Hives (urticaria) are raised, itchy welts triggered by allergens, stress, or infection.",
    "Cellulitis is a bacterial skin infection causing redness, warmth, and swelling; requires antibiotic treatment.",
    "Acne vulgaris is caused by clogged hair follicles; treatment includes topical retinoids and benzoyl peroxide.",
    # ── Musculoskeletal / Ortho ───────────────────────────────────────────────
    "Osteoarthritis is the most common joint disease causing pain and stiffness, especially in knees and hips.",
    "Rheumatoid arthritis is an autoimmune disease causing symmetrical joint inflammation, stiffness, and swelling.",
    "Sciatica is nerve pain radiating from the lower back through the buttock and down one leg.",
    "A herniated disc occurs when the soft disc material protrudes and compresses nearby nerves causing pain.",
    "Muscle strains from overuse heal with RICE: Rest, Ice, Compression, and Elevation.",
    # ── Diabetes / Endocrine ──────────────────────────────────────────────────
    "Type 1 diabetes is an autoimmune condition where the pancreas produces no insulin; requires daily insulin.",
    "Type 2 diabetes is caused by insulin resistance; managed with lifestyle changes, oral medications, and insulin.",
    "Hypoglycemia (low blood sugar) causes trembling, sweating, confusion; treat immediately with glucose or juice.",
    "Hypothyroidism causes fatigue, weight gain, cold intolerance, and constipation due to low thyroid hormone.",
    "Hyperthyroidism (overactive thyroid) causes weight loss, rapid heartbeat, sweating, and anxiety.",
    # ── Mental Health ─────────────────────────────────────────────────────────
    "Major depressive disorder causes persistent sadness, loss of interest, sleep disturbance, and fatigue.",
    "Generalized anxiety disorder involves excessive, uncontrollable worry about everyday events for 6+ months.",
    "Panic attacks cause sudden intense fear, chest pain, shortness of breath, and dizziness lasting minutes.",
    "Insomnia is difficulty falling or staying asleep; sleep hygiene and CBT-I are first-line treatments.",
    "PTSD develops after traumatic events; symptoms include flashbacks, nightmares, and hypervigilance.",
    # ── ENT ───────────────────────────────────────────────────────────────────
    "Sinusitis is inflammation of the sinuses causing facial pain, nasal congestion, and thick nasal discharge.",
    "Allergic rhinitis (hay fever) causes sneezing, runny nose, and itchy eyes triggered by allergens.",
    "Otitis media (middle ear infection) is common in children; causes ear pain and hearing loss.",
    "Tonsillitis causes sore throat, swollen tonsils, fever, and difficulty swallowing.",
    "Vertigo is a sensation of spinning often caused by inner ear disorders like BPPV or Meniere's disease.",
    # ── Eye / Ophthalmology ───────────────────────────────────────────────────
    "Conjunctivitis (pink eye) causes redness, discharge, and itching; can be viral, bacterial, or allergic.",
    "Glaucoma is increased intraocular pressure that damages the optic nerve; a leading cause of blindness.",
    "Diabetic retinopathy is damage to retinal blood vessels caused by long-term high blood sugar levels.",
    "Sudden vision loss, eye pain, or flashing lights require immediate ophthalmological evaluation.",
    # ── Kidney / Urology ──────────────────────────────────────────────────────
    "Urinary tract infections (UTIs) cause burning urination, urgency, and cloudy urine; treated with antibiotics.",
    "Kidney stones cause severe colicky flank pain that radiates to the groin; most pass spontaneously.",
    "Chronic kidney disease progresses silently; early signs include fatigue, swelling, and reduced urine output.",
    "Benign prostatic hyperplasia (BPH) causes urinary hesitancy, weak stream, and frequent nighttime urination.",
    # ── Women's Health ────────────────────────────────────────────────────────
    "Polycystic ovary syndrome (PCOS) causes irregular periods, acne, hair growth, and potential infertility.",
    "Endometriosis causes severe menstrual pain and can lead to infertility if untreated.",
    "Postpartum depression affects up to 15% of new mothers and requires professional support and treatment.",
    # ── Pediatric ─────────────────────────────────────────────────────────────
    "Fever in infants under 3 months above 38°C (100.4°F) is a medical emergency requiring immediate evaluation.",
    "RSV (respiratory syncytial virus) is the leading cause of bronchiolitis in infants and young children.",
    "Hand, foot, and mouth disease is a common viral illness in children causing sores and rash.",
    # ── Oncology ──────────────────────────────────────────────────────────────
    "Warning signs of cancer include unexplained weight loss, persistent fatigue, lumps, or blood in stool/urine.",
    "Early detection through screening (mammograms, colonoscopy, PSA) significantly improves cancer outcomes.",
    # ── Nutrition / Lifestyle ─────────────────────────────────────────────────
    "Anemia (iron deficiency) causes fatigue, pale skin, and shortness of breath; treated with iron supplements.",
    "Dehydration symptoms include dark urine, dizziness, dry mouth; adults should drink 2–3 litres of water daily.",
    "Obesity is a risk factor for type 2 diabetes, heart disease, sleep apnea, and certain cancers.",
    "Regular physical activity of 150 minutes per week reduces risk of cardiovascular disease and depression.",
    "A balanced diet rich in fruits, vegetables, whole grains, and lean protein supports immune function.",
    # ── Emergency Signs ───────────────────────────────────────────────────────
    "Signs of stroke: Face drooping, Arm weakness, Speech difficulty — act FAST and call emergency services.",
    "Signs of anaphylaxis: throat swelling, hives, drop in blood pressure; administer epinephrine immediately.",
    "Loss of consciousness, unresponsiveness, or severe breathing difficulty always require a 112/911 call.",
    "Sepsis warning signs: high fever or very low temperature, rapid heart rate, confusion, and low blood pressure.",
]


def _build_rag_index():
    """
    Build a FAISS flat-L2 index from MEDICAL_KB using sentence-transformers.
    Called once at container startup — takes ~3–5 seconds on first run.
    Returns (index, embedder, kb_list) so chat() can retrieve at query time.
    """
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 80 MB, CPU-only
    embeddings = embedder.encode(MEDICAL_KB, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)  # Inner-product == cosine on normalised vectors
    index.add(embeddings.astype("float32"))
    return index, embedder, MEDICAL_KB

EMERGENCY_FOLLOWUP = {
    "Cardiologist":      ("Do not move — sit or lie down quietly and loosen any tight clothing.", "Is there someone with you who can call 112 or 911 right now?", "Are you feeling increasing chest pressure or spreading pain to your jaw or arm?"),
    "Pulmonologist":     ("Sit upright, open windows for fresh air, and do not lie flat.", "Is someone with you who can call emergency services immediately?", "Is the difficulty breathing getting worse every few minutes?"),
    "Neurologist":       ("Do not give the patient anything to eat or drink — keep them still and calm.", "Is someone with you right now to call 112 or 911?", "Are symptoms like numbness or vision loss getting worse rapidly?"),
    "General Physician": ("Stay calm, sit down, and do not attempt to drive yourself to hospital.", "Is there someone nearby who can call emergency services for you?", "Are any of your symptoms rapidly getting worse right now?"),
}

def get_emergency_followup(specialist: str) -> tuple:
    return EMERGENCY_FOLLOWUP.get(specialist, EMERGENCY_FOLLOWUP["General Physician"])


def _detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("groq-secret")],
    max_containers=100,
)
@modal.concurrent(max_inputs=20)
class GroqMedical:

    @modal.enter()
    def setup(self):
        import os
        self.groq_api_key = os.environ["GROQ_API_KEY"]
        # ── RAG: build index once per container (cold-start only) ──
        self.rag_index, self.rag_embedder, self.rag_kb = _build_rag_index()

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Embed the user query, search FAISS, and return the top-k
        medical KB facts as a formatted context block.
        """
        import numpy as np
        q_vec = self.rag_embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = self.rag_index.search(q_vec.astype("float32"), top_k)
        retrieved = [self.rag_kb[i] for i in indices[0] if i < len(self.rag_kb)]
        if not retrieved:
            return ""
        context_lines = "\n".join(f"- {fact}" for fact in retrieved)
        return (
            "[Relevant Medical Context Retrieved via RAG]\n"
            f"{context_lines}\n"
            "[Use the above context to ground your response. Do not copy it verbatim.]\n\n"
        )

    def _groq_chat(self, text: str, max_tokens: int = 250, image_b64: str = "") -> str:
        """Call Groq's OpenAI-compatible chat completions endpoint."""
        import requests, base64

        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            mime = _detect_image_mime(img_bytes)
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ]
        else:
            content = text

        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "reasoning_effort": "none",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["choices"][0]["message"]["content"].strip()
        result = result.replace("```", "").strip()
        for stop in ["<|im_end|>", "<|im_start|>", "\nUser:", "User:",
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

        raw = self._groq_chat(classify_prompt, max_tokens=10).upper().strip()
        if "LEVEL3" in raw:
            return "LEVEL3"
        elif "LEVEL2" in raw:
            return "LEVEL2"
        return "LEVEL1"

    def _classify_continuation(self, prev_complaint: str, prev_keywords: str, message: str) -> str:
        """Return 'NEW' or 'CONTINUE' — does this message start a new problem
        or continue the current one? Defaults to CONTINUE on any failure."""
        prompt = CONTINUATION_PROMPT.format(
            prev=prev_complaint or prev_keywords or "a medical complaint",
            keywords=prev_keywords or "n/a",
            message=message,
        )
        try:
            raw = self._groq_chat(prompt, max_tokens=5).upper()
        except Exception:
            return "CONTINUE"
        return "NEW" if "NEW" in raw else "CONTINUE"

    @modal.method()
    def chat(self, user_id: str, message: str, image_b64: str = "") -> dict:
        import time

        if not image_b64 and is_greeting(message):
            return {
                "response": GREETING_REPLY,
                "level": "GREETING",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
                "follow_up_questions": [],
            }

        if any(kw in message.lower() for kw in SELF_HARM_WORDS):
            return {
                "response": CRISIS_RESPONSE,
                "level": "CRISIS",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
                "follow_up_questions": [],
            }

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

        image_observation = ""
        if image_b64:
            try:
                image_observation = self._groq_chat(
                    f"{IMAGE_ANALYSIS_PROMPT}\n\n{message}", max_tokens=100, image_b64=image_b64
                )
                message = f"{message} [Image: {image_observation}]"
            except Exception:
                # Image unusable (too large / unsupported format for Groq) —
                # drop it and continue text-only instead of failing the chat.
                image_observation = ""
                image_b64 = ""

        # ── RAG: retrieve grounding context before severity detection ──
        rag_context = self._retrieve_context(message)

        start = time.time()
        severity = self._detect_severity(message)
        # Safety net: hard red-flag phrases always escalate to emergency,
        # regardless of what the classifier decided.
        if any(kw in message.lower() for kw in EMERGENCY_OVERRIDE):
            severity = "LEVEL3"

        prev_data       = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": ""})
        prev_keywords   = prev_data.get("keywords", "")
        prev_specialist = prev_data.get("specialist", "")

        is_hinglish = detect_hinglish(message)
        lang_note = f"\n\n{HINGLISH_INSTRUCTION}" if is_hinglish else ""

        # Chat 1: detect fresh. Chat 2/3: ask the model (semantically, using the
        # previous complaint as context) whether this CONTINUES the current
        # problem or starts a NEW one — so extra details like "ye gardan pe hai"
        # (rash location) keep the same specialist instead of flipping.
        prev_complaint = prev_data.get("complaint", "")
        if count == 0:
            topic_changed = False
            specialist = detect_specialist(message)
        else:
            topic_changed = self._classify_continuation(prev_complaint, prev_keywords, message) == "NEW"
            if topic_changed:
                specialist = detect_specialist(message)
            else:
                specialist = prev_specialist or detect_specialist(prev_keywords) or detect_specialist(message)

        # ----------------------------------------------------------------
        # CHAT 1 (or topic switch) — full one-shot triage response
        # ----------------------------------------------------------------
        if count == 0 or topic_changed:
            if severity == "LEVEL1":
                system = rag_context + LEVEL_1_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note
            elif severity == "LEVEL2":
                system = rag_context + LEVEL_2_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note
            else:
                system = rag_context + LEVEL_3_PROMPT.replace("{specialist}", specialist).replace("{user_message}", message) + lang_note

            raw = self._groq_chat(f"{system}\n\n{message}", max_tokens=400, image_b64=image_b64)
            if not raw.strip():
                # Rare transient empty completion — retry once before falling through.
                raw = self._groq_chat(f"{system}\n\n{message}", max_tokens=400, image_b64=image_b64)

            if not raw.strip().startswith("- "):
                lines = raw.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("- "):
                        raw = "\n".join(lines[i:])
                        break

            DASH_PREFIXES = ["Mild Concern", "Moderate Concern", "EMERGENCY",
                             "Instruction:", "General Advice:", "Consult:",
                             "Keywords:", "Q1:", "Q2:"]
            cleaned_lines = []
            for line in raw.split("\n"):
                s = line.strip()
                if s and not s.startswith("- "):
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

            kw_for_q = ""
            for line in response.split("\n"):
                s = line.strip().lstrip("- ")
                if s.startswith("Keywords:"):
                    kw_for_q = s.replace("Keywords:", "").strip()
                    break
            if not kw_for_q:
                kw_for_q = message

            def fmt(prompt, **kwargs):
                for k, v in kwargs.items():
                    prompt = prompt.replace("{" + k + "}", v)
                return prompt

            try:
                q1_raw = self._groq_chat(
                    fmt(FOLLOWUP_Q1_PROMPT, specialist=specialist, keywords=kw_for_q, message=message, lang_note=lang_note),
                    max_tokens=60
                )
                q1_text = q1_raw.split("\n")[0].strip()
            except Exception:
                q1_text = ""
            if not q1_text:
                q1_text = "Yeh symptoms kitne time se hain aur kitna severe hai?" if is_hinglish else "Can you describe when the symptoms started and how severe they are?"

            try:
                q2_raw = self._groq_chat(
                    fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw_for_q, message=message, lang_note=lang_note),
                    max_tokens=60
                )
                q2_text = q2_raw.split("\n")[0].strip()
            except Exception:
                q2_text = ""
            if not q2_text:
                q2_text = "Aur koi body part bhi affected hai kya?" if is_hinglish else "Have you noticed any other body areas showing similar symptoms?"

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
                    f"- Consult: [{specialist}]\n"
                    f"- Immediate Action: {advice}\n"
                    f"- Keywords: {kw}\n"
                    f"- EMERGENCY: Call 112 (India) or 911 (US) NOW or go to the nearest hospital.\n"
                    f"- Q1: {q1}\n"
                    f"- Q2: {q2}"
                )
            else:
                def fmt(prompt, **kwargs):
                    for k, v in kwargs.items():
                        prompt = prompt.replace("{" + k + "}", v)
                    return prompt

                advice_raw = self._groq_chat(
                    fmt(FOLLOWUP_ADVICE_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    max_tokens=80, image_b64=image_b64
                )
                advice_text = strip_medicines(advice_raw.split("\n")[0].strip())
                if not advice_text:
                    advice_text = "Aaram karein aur paani zyada piyein" if is_hinglish else "Rest and avoid strenuous activity; stay well hydrated"

                q1_raw = self._groq_chat(
                    fmt(FOLLOWUP_Q1_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    max_tokens=60
                )
                q1_text = q1_raw.split("\n")[0].strip()
                if not q1_text:
                    q1_text = "Yeh problem kitne time se hai aur severity kya hai?" if is_hinglish else "Can you describe how your symptom has changed since it started?"

                q2_raw = self._groq_chat(
                    fmt(FOLLOWUP_Q2_PROMPT, specialist=specialist, keywords=kw, message=message, lang_note=lang_note),
                    max_tokens=60
                )
                q2_text = q2_raw.split("\n")[0].strip()
                if not q2_text:
                    q2_text = "Aur koi body part bhi affected hai kya?" if is_hinglish else "Have you noticed any other symptoms appearing alongside this?"

                # Same labeled format as Chat 1 so the frontend renders every
                # chat as the same structured card.
                response = (
                    f"- Consult: [{specialist}]\n"
                    f"- General Advice: {advice_text}\n"
                    f"- Keywords: {kw}\n"
                    f"- Q1: {q1_text}\n"
                    f"- Q2: {q2_text}"
                )

                if image_observation:
                    response = f"- Image Observation: {image_observation}\n" + response

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
        session_ended = new_count >= MAX_CHATS_PER_SESSION

        session_data = sessions.get(f"{user_id}_record", {"chats": {}, "keywords": "", "specialist": "", "complaint": ""})
        session_data["keywords"]   = keywords
        session_data["specialist"] = specialist
        if count == 0 or topic_changed:
            session_data["complaint"] = message[:200]
        session_data["chats"][f"query_{new_count}"] = {
            "input":  {"query_text": message, "has_image": bool(image_b64)},
            "output": {"text": response}
        }
        sessions[f"{user_id}_record"] = session_data

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
        try:
            return self.chat.local(user_id, message, image_b64)
        except Exception as e:
            # Never surface a raw 500 — return a friendly, retry-able message.
            return {
                "response": "Sorry, something went wrong on our side. Please try sending that again in a moment.",
                "level": "ERROR",
                "session_count": None,
                "session_ended": False,
                "specialist": None,
                "follow_up_questions": [],
                "error": str(e)[:200],
            }
