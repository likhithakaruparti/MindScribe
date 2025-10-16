from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForSequenceClassification

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="MindScribe v2.0 - Fully Local")

# CORS middleware (so frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev: allow any origin
    allow_credentials=True,
    allow_methods=["*"],   # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# -----------------------
# Request models
# -----------------------
class TextInput(BaseModel):
    text: str

class MoodUpdate(BaseModel):
    timestamp: str
    user_mood: str
    user_emoji: str

class TaskUpdate(BaseModel):
    timestamp: str
    task_index: int
    done: bool

# -----------------------
# Storage
# -----------------------
STORAGE_FILE = "entries.json"

def save_entry(entry):
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------
# Local LLM pipeline (flan-t5-small)
# -----------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# -----------------------
# Local emotion classifier
# -----------------------
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, return_all_scores=True)

EMOJI_MAP = {
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "sadness": "😢",
    "fear": "😱",
    "surprise": "😲",
    "neutral": "😐"
}

# -----------------------
# Helpers
# -----------------------
def detect_emotion(text: str):
    """Detect emotion using local pipeline"""
    try:
        results = emotion_pipeline(text)[0]  # list of dicts with 'label' & 'score'
        top_emotion = max(results, key=lambda x: x['score'])
        label = top_emotion['label'].lower()
        emoji = EMOJI_MAP.get(label, "😐")
        return label, emoji
    except:
        return "neutral", "😐"

def generate_ai_output(text: str, emotion: str):
    """Generate summary, tasks, highlights, suggestions using local LLM"""
    prompt = f"""
You are an assistant that processes a diary entry. 
From the given text:
1. Provide a short summary.
2. Extract tasks (things the user wants/should do).
3. Extract highlights (achievements or completed actions).
4. Give 2-3 personalized suggestions based on:
   - user's mood
   - tasks done or not
   - highlights
Text: {text}
Detected mood: {emotion}
Return as JSON with keys: summary, tasks (list), highlights (list), suggestions (list of strings).
"""
    output = llm_pipeline(prompt, max_length=512, do_sample=True)[0]['generated_text']
    try:
        return json.loads(output)
    except:
        # fallback if parsing fails
        sentences = text.split(".")
        summary = ". ".join(sentences).strip()
        tasks = [s.strip() for s in sentences if "want" in s or "plan" in s or "should" in s]
        highlights = [s.strip() for s in sentences if "managed" in s or "finished" in s or "completed" in s]
        suggestions = []
        if emotion in ["sadness", "anger", "fear"]:
            suggestions.append("Take a short meditation or break.")
        if tasks and all(t.lower() for t in tasks):
            suggestions.append("Congrats! You completed all tasks today!")
        if not highlights:
            suggestions.append("Try noting at least one small win today!")
        return {
            "summary": summary,
            "tasks": tasks,
            "highlights": highlights,
            "suggestions": suggestions
        }

# -----------------------
# Endpoints
# -----------------------
@app.get("/")
def home():
    return {"message": "MindScribe backend is running 🚀"}

@app.post("/process_entry/")
def process_entry(input: TextInput):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Detect emotion locally
        emotion, emoji = detect_emotion(input.text)

        # Generate summary, tasks, highlights, suggestions
        ai_output = generate_ai_output(input.text, emotion)
        tasks_with_status = [{"task": t, "done": False} for t in ai_output.get("tasks", [])]

        entry = {
            "timestamp": timestamp,
            "text": input.text,
            "summary": ai_output.get("summary", ""),
            "tasks": tasks_with_status,
            "highlights": ai_output.get("highlights", []),
            "emotion": emotion,
            "emoji": emoji,
            "user_mood": emotion,
            "user_emoji": emoji,
            "suggestions": ai_output.get("suggestions", [])
        }

        save_entry(entry)
        return entry
    except Exception as e:
        return {"error": str(e)}

@app.get("/entries/")
def get_entries():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE) as f:
            return json.load(f)
    return []

@app.post("/update_mood/")
def update_mood(update: MoodUpdate):
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            data = json.load(f)
    else:
        return {"error": "No entries found"}

    for entry in data:
        if entry["timestamp"] == update.timestamp:
            entry["user_mood"] = update.user_mood
            entry["user_emoji"] = update.user_emoji
            break
    else:
        return {"error": "Entry not found"}

    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return {"message": "Mood updated successfully"}

@app.post("/update_task/")
def update_task(update: TaskUpdate):
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            data = json.load(f)
    else:
        return {"error": "No entries found"}

    for entry in data:
        if entry["timestamp"] == update.timestamp:
            if 0 <= update.task_index < len(entry["tasks"]):
                entry["tasks"][update.task_index]["done"] = update.done
                break
            else:
                return {"error": "Invalid task index"}
    else:
        return {"error": "Entry not found"}

    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return {"message": "Task updated successfully"}
