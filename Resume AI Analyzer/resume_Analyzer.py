import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1. Extract text from resume PDF
# -------------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------------------------
# 2. Preprocess text using spaCy
# -------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# -------------------------
# 3. Calculate similarity score
# -------------------------
def get_match_score(resume_text, job_text):
    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_text)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean, job_clean])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    score = round(score * 100, 2)

    job_keywords = set(job_clean.split())
    matched_keywords = [word for word in job_keywords if word in resume_clean]

    return score, matched_keywords

# -------------------------
# 4. Main function
# -------------------------
def main():
    resume_path = "sample_resume.pdf"
    job_path = "job_description.txt"

    print("📄 Reading resume and job description...")

    resume_text = extract_text_from_pdf(resume_path)
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_text = f.read()

    score, keywords = get_match_score(resume_text, job_text)

    print("\n✅ Analysis Complete!")
    print("🎯 Resume Match Score:", score, "%")
    print("🔑 Matched Keywords:")
    if keywords:
        print(", ".join(keywords))
    else:
        print("No keywords matched.")

# -------------------------
# Run it
# -------------------------
if __name__ == "__main__":
    main()
