
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Load the local model and tokenizer
LOCAL_MODEL_PATH = "E:/笔记/ai模型/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)

# Translation function
def en2zh_translate(english_text, max_new_tokens=200):
    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    chinese_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return chinese_text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text=r""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Route to upload and translate PDF
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        pdf_file = request.files.get('pdf_file')
        if pdf_file:
            # Save the uploaded file
            pdf_path = f"./uploaded_{pdf_file.filename}"
            pdf_file.save(pdf_path)

            # Extract text from PDF
            english_text = extract_text_from_pdf(pdf_path)

            # Translate the text
            chinese_text = en2zh_translate(english_text)

            # Save the translated text to a file
            translated_file_path = "./translated_text.txt"
            with open(translated_file_path, "w", encoding="utf-8") as f:
                f.write(chinese_text)

            return render_template('index_pdf.html', message="Translation completed! Check 'translated_text.txt'.")
    return render_template('index_pdf.html')

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
