from flask import Flask, render_template, request
import spacy
import fitz  # PyMuPDF
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load skills list
with open("skills.txt", "r") as f:
    SKILLS = [s.strip().lower() for s in f.readlines()]

# -------------------- TEXT EXTRACTION --------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text.lower()

# -------------------- SKILL EXTRACTION --------------------
def extract_skills(text):
    skill_freq = {}
    for skill in SKILLS:
        count = text.count(skill)
        if count > 0:
            skill_freq[skill] = count
    return skill_freq

# -------------------- SKILL CONFIDENCE --------------------
def skill_confidence(skill_freq):
    confidence = {}
    for skill, freq in skill_freq.items():
        if freq >= 3:
            confidence[skill] = ("High", 3)
        elif freq == 2:
            confidence[skill] = ("Medium", 2)
        else:
            confidence[skill] = ("Low", 1)
    return confidence

# -------------------- RESUME–JD MATCHING --------------------
def resume_jd_match(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["resume"]
        jd_text = request.form["job_description"]

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)

        # Skill extraction + confidence
        skill_freq = extract_skills(resume_text)
        skill_conf = skill_confidence(skill_freq)

        # Resume–JD match score
        match_score = resume_jd_match(resume_text, jd_text)

        # Matching & missing skills
        resume_skills = set(skill_freq.keys())
        jd_skills = {skill for skill in SKILLS if skill in jd_text.lower()}

        matching_skills = sorted(resume_skills & jd_skills)
        missing_skills = sorted(jd_skills - resume_skills)

        return render_template(
            "result.html",
            skill_conf=skill_conf,
            match_score=match_score,
            matching_skills=matching_skills,
            missing_skills=missing_skills
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
