from typing import Tuple, List
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi import FastAPI, File, UploadFile, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import io
import re
import datetime
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
app = FastAPI()

#Allow frontend JS to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup (SQLite)
conn = sqlite3.connect("score.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        users_id INTEGER,
        job_id INTEGER,
        filename TEXT,
        score REAL,
        explanation TEXT,
        name TEXT,
        contact TEXT,
        uploaded_at TEXT,
        FOREIGN KEY (users_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE SET NULL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,  
        email TEXT UNIQUE,
        created_at TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        job_name TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
""")

conn.commit()

# Extract PDF text
def extract_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    return text

def split_into_sentences(text):
    # Split by period, exclamation, question marks + trim whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out short/empty ones
    return [s for s in sentences if len(s.split()) > 3]

import re

def extract_contact_info(text: str) -> Tuple[str, str]:
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "N/A"

    # Extract phone number
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
    phone = phone_match.group(0) if phone_match else "N/A"

    # Extract probable name from the first few lines (not guaranteed)
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if len(l.strip().split()) <= 4 and len(l.strip()) > 3]
    name = lines[0] if lines else "N/A"

    return name, f"{email} | {phone}"


def calculate_score(resume: str, jd: str) -> Tuple[float, str]:
    resume_sentences = split_into_sentences(resume)
    jd_sentences = split_into_sentences(jd)

    if not resume_sentences or not jd_sentences:
        return 0.0, "Could not extract meaningful sentences from resume or job description."

    # Encode all at once (faster)
    resume_embeddings = model.encode(resume_sentences)
    jd_embeddings = model.encode(jd_sentences)

    # For each JD sentence, find best matching resume sentence
    match_scores = []
    matched_sentences = []

    for i, jd_emb in enumerate(jd_embeddings):
        similarities = cosine_similarity([jd_emb], resume_embeddings)[0]
        best_score = max(similarities)
        match_scores.append(best_score)

        best_idx = similarities.argmax()
        matched_sentences.append((jd_sentences[i], resume_sentences[best_idx], best_score))

    # Calculate final score (average of top matches)
    final_score = sum(match_scores) / len(match_scores)

    # Create explanation with best matches
    top_matches = sorted(matched_sentences, key=lambda x: x[2], reverse=True)[:5]
    explanation_lines = [
        f"JD: {jd_line}\nResume Match: {res_line} (Score: {round(score * 100, 1)}%)"
        for jd_line, res_line, score in top_matches
    ]
    explanation = (
        f"The resume has a {round(final_score * 100, 2)}% alignment with the job description based on semantic similarity.\n\n"
        "Top Matches:\n" + "\n\n".join(explanation_lines)
    )

    return final_score, explanation


#login credentials
@app.post("/login")
async def login(username: str=Form(...), password: str=(Form(...))):
    cursor.execute("SELECT id from users WHERE username=? AND password=?", (username,password))
    user=cursor.fetchone()
    if user:
        user_id=user[0]
        return JSONResponse(content={"user_id": user_id, "message": "Login successful"})
    else:
         return JSONResponse(content={"message": "Invalid credentials"}, status_code=401)
    
#create Account Credentials
@app.post("/create")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None)  # Optional for now
):
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email, created_at) VALUES (?, ?, ?, ?)",
            (username, password, email, datetime.datetime.now().isoformat())
        )
        conn.commit()
        return RedirectResponse("/login.html", status_code=303)  # Redirect after success
    except sqlite3.IntegrityError:
        return HTMLResponse("<h3>Username or email already exists. Please go back and try again.</h3>")


# Upload & Score Endpoint
@app.post("/score")
async def score_resume(user_id: int=Form(...), resume: UploadFile = File(...), jd: UploadFile = File(...)):
    
    resume_bytes = await resume.read()
    jd_bytes = await jd.read()

    resume_text = extract_text(resume_bytes)
    jd_text = extract_text(jd_bytes)

    score, explanation = calculate_score(resume_text, jd_text)
    name, contact = extract_contact_info(resume_text)

    # Save to database
    cursor.execute(
        "INSERT INTO resumes (users_id, filename, score, explanation, name, contact, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, resume.filename, round(score * 100, 2), explanation, name, contact, datetime.datetime.now().isoformat())
    )
    conn.commit()

    return {"filename": resume.filename, "score": round(score * 100, 2), "explanation": explanation}

@app.post("/score-multiple")
async def score_multiple_resumes(
    user_id: int=Form(...),
    job_name: str = Form(...),
    jd: UploadFile = File(...),
    resumes: List[UploadFile] = File(...)
):
    jd_bytes = await jd.read()
    jd_text = extract_text(jd_bytes)

    cursor.execute(
        "INSERT INTO jobs (user_id, job_name, created_at) VALUES (?, ?, ?)",
        (user_id, job_name, datetime.datetime.now().isoformat())
    )
    job_id = cursor.lastrowid

    results = []

    for i, resume_file in enumerate(resumes, start=1):
        resume_bytes = await resume_file.read()
        resume_text = extract_text(resume_bytes)

        score, explanation = calculate_score(resume_text, jd_text)
        name, contact = extract_contact_info(resume_text)

        # Save to DB
        cursor.execute(
            "INSERT INTO resumes (users_id, job_id, filename, score, explanation, name, contact, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, job_id, resume_file.filename, round(score * 100, 2), explanation, name, contact, datetime.datetime.now().isoformat())
        )
        conn.commit()

        results.append({
        "number": i,
        "filename": resume_file.filename,
        "score": round(score * 100, 2),
        "explanation": explanation,
        "name": name,
        "contact": contact
        })

    return results

# Get sorted resumes
@app.get("/resumes")
def get_resumes(user_id: int = Query(...)):
    cursor.execute("SELECT filename, score, explanation, name, contact, uploaded_at FROM resumes WHERE users_id=? ORDER BY score DESC", (user_id,))
    rows = cursor.fetchall()
    return [{"filename": r[0], "score": r[1], "explanation": r[2], "name": r[3], "contact": r[4], "uploaded_at": r[5]} for r in rows]

@app.get("/users")
def get_resumes(user_id: int = Query(...)):
    cursor.execute("SELECT id, username, password, email FROM users WHERE id=?", (user_id,))
    rows = cursor.fetchall()
    return [{"id": r[0], "username": r[1], "password": r[2], "email": r[3]} for r in rows]

@app.get("/user-jobs")
def get_user_jobs(user_id: int = Query(...)):
    cursor.execute("SELECT id, job_name FROM jobs WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    return [{"job_id": r[0], "job_name": r[1]} for r in rows]

@app.get("/resumes-by-job")
def get_resumes_by_job(user_id: int = Query(...), job_id: int = Query(...)):
    cursor.execute("""
        SELECT filename, score, explanation, name, contact, uploaded_at
        FROM resumes
        WHERE users_id=? AND job_id=?
        ORDER BY score DESC
    """, (user_id, job_id))
    rows = cursor.fetchall()
    return [{"filename": r[0], "score": r[1], "explanation": r[2], "name": r[3], "contact": r[4], "uploaded_at": r[5]} for r in rows]

@app.put("/rename-job/{job_id}")
def rename_job(job_id: int, new_name: str = Body(...)):
    cursor.execute("UPDATE jobs SET job_name=? WHERE id=?", (new_name, job_id))
    conn.commit()
    return {"message": "Job renamed successfully"}

@app.delete("/delete-job/{job_id}")
def delete_job(job_id: int):
    cursor.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    return {"message": "Job deleted successfully"}


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use 8000 locally, Render’s port in deployment
    uvicorn.run("aiCheck:app", host="0.0.0.0", port=port)