
from flask import Flask, request, jsonify
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

# RESTful API endpoint for translation
@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid input, 'text' field is required"}), 400

        # Perform translation
        english_text = data['text']
        chinese_text = en2zh_translate(english_text)

        # Return the result
        return jsonify({"english_text": english_text, "chinese_text": chinese_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
