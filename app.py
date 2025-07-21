import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for servers

import matplotlib.pyplot as plt
import os
import spacy
import matplotlib.pyplot as plt
import textstat
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import PyPDF2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

nlp = spacy.load('en_core_web_sm')

FIXED_KEYWORDS = [
    "python", "java", "sql", 
    "machine learning", "data analysis", 
    "communication", "teamwork"
]

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        text = "Error reading PDF"
    return text

def analyze_resume(text, user_keywords):
    doc = nlp(text.lower())
    all_keywords = list(set(FIXED_KEYWORDS + user_keywords))
    keywords_count = {key: 0 for key in all_keywords if key.strip()}

    for token in doc:
        for key in keywords_count:
            if key in token.lemma_:
                keywords_count[key] += 1

    return keywords_count

def calculate_readability(text):
    try:
        score = textstat.flesch_reading_ease(text)
    except:
        score = 0.0
    return round(score, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    keyword_scores = {}
    chart_url = None
    filename = ""
    resume_text = ""
    readability_score = 0.0
    resume_length = 0
    user_keywords = []

    if request.method == 'POST':
        uploaded_file = request.files['resume']
        user_keywords_input = request.form.get('user_keywords', '')
        user_keywords = [kw.strip().lower() for kw in user_keywords_input.split(',') if kw.strip()]

        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            resume_text = extract_text_from_pdf(filepath)
            resume_length = len(resume_text.split())

            keyword_scores = analyze_resume(resume_text, user_keywords)
            readability_score = calculate_readability(resume_text)

            # Plot chart
            plt.figure(figsize=(12, 6))
            plt.bar(keyword_scores.keys(), keyword_scores.values(), color='mediumseagreen')
            plt.title('Resume Keyword Match Score (Fixed + Custom)')
            plt.xlabel('Skills / Keywords')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_path = os.path.join(STATIC_FOLDER, 'chart.png')
            plt.savefig(chart_path)
            plt.close()

            chart_url = 'chart.png'

            # Generate TXT report
            with open('static/report.txt', 'w') as f:
                f.write(f"Resume File: {filename}\n")
                f.write(f"Resume Length (in words): {resume_length}\n")
                f.write(f"Readability Score: {readability_score}\n\n")
                f.write("Keyword Frequency (Fixed + Custom):\n")
                for key, value in keyword_scores.items():
                    f.write(f"{key}: {value}\n")

    return render_template(
        'index.html',
        chart_url=chart_url,
        keywords=keyword_scores,
        fixed_keywords=FIXED_KEYWORDS,
        user_keywords=user_keywords,
        filename=filename,
        resume_length=resume_length,
        readability_score=readability_score
    )

@app.route('/download-report')
def download_report():
    return send_file('static/report.txt', as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
