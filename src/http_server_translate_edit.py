
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# Route to render the HTML page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input text from the form
        english_text = request.form.get('text', '')
        chinese_text = en2zh_translate(english_text) if english_text else ''
        return render_template('index.html', english_text=english_text, chinese_text=chinese_text)
    return render_template('index.html')

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
