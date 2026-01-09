# Import Libraries
from flask import Flask, render_template, request       # Flask web framework for backend
from werkzeug.utils import secure_filename              # For safely saving uploaded files
import os  
import PyPDF2                                           # For extracting text from PDF files
import docx                                             # For extracting text from DOCX files (from python-docx library)
from transformers import pipeline                       # Hugging Face Transformers pipeline for NLP tasks (QA)



# Initialize Flask App
app = Flask(__name__)


# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt", "docx"}

# Load QA Model (HuggingFace LLM Model - DistilBERT)
qa_model = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def extract_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    # Remove excessive newlines
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = " ".join(text.split())

    return text

def extract_docx_text(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_txt_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_sentences(text):
    return [s.strip() for s in text.split(".") if len(s.strip()) > 30]


def highlight_answer(sentence, answer):
    return sentence.replace(answer, f"<mark>{answer}</mark>")


# Main Route
@app.route("/", methods=["GET", "POST"])
def home():
    answers = []

    if request.method == "POST":

        question = request.form.get("question", "").strip()
        text_input = request.form.get("text_input", "").strip()
        files = request.files.getlist("files")
        # Determine input source
        if files and any(f.filename for f in files):
            input_source = "Uploaded Document"
        elif text_input:
            input_source = "Direct Text Input"
        else:
            input_source = "Unknown Source"


        combined_text = ""

        # Handle uploaded files
        for f in files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)

                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                f.save(path)

                ext = filename.rsplit(".", 1)[1].lower()
                if ext == "pdf":
                    combined_text += clean_text(extract_pdf_text(path))
                elif ext == "docx":
                    combined_text += clean_text(extract_docx_text(path))
                elif ext == "txt":
                    combined_text += clean_text(extract_txt_text(path))

        # Add direct text input
        if text_input:
            combined_text += text_input

        # Validation
        if not combined_text.strip():
            answers = [{
                "text": "No document or text input provided.",
                "confidence": 0.0,
                "source": "-"
            }]
        elif not question:
            answers = [{
                "text": "Please enter a question.",
                "confidence": 0.0,
                "source": "-"
            }]
        else:
            
            # Extractive QA Logic
            sentences = split_into_sentences(combined_text)
            answers = []

            for sent in sentences[:15]:
                result = qa_model(question=question, context=sent)

                if result["score"] > 0.2:
                    answers.append({
                        "text": sent.replace(
                            result["answer"],
                            f"<mark>{result['answer']}</mark>"
                        ),
                        "confidence": round(result["score"], 2),
                        "source": input_source
                    })

            # Sort by confidence and keep top 3
            answers = sorted(answers, key=lambda x: x["confidence"], reverse=True)
            answers = answers[:3]

            # Fallback
            if not answers:
                answers = [{
                    "text": "No relevant answer found.",
                    "confidence": 0.0,
                    "source": "-"
                }]


    print("ANSWERS SENT TO UI:", answers)
    return render_template("index.html", answers=answers)


# Run App
if __name__ == "__main__":
    app.run(debug=True)
