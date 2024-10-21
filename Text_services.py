from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import language_tool_python
from googletrans import Translator
import requests
import os

app = Flask(__name__)

# Initialize summarization model
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# Grammar check function using LanguageTool
def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')  # Set the language to English (US)
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    corrections = [{'error': match.ruleId, 'message': match.message} for match in matches]  # List corrections
    return corrected_text, corrections

# Plagiarism checker (using a mock API as example)
def check_plagiarism(text):
    try:
        api_url = "https://plagiarism-api-example.com/check"  # Replace with actual API
        data = {'text': text}
        response = requests.post(api_url, json=data)
        result = response.json()  # Mock API response
        return result.get('plagiarism_percentage', 0), result.get('plagiarism_report', 'No report available.')
    except Exception as e:
        return 0, f"Error in plagiarism check: {str(e)}"

# Translator using googletrans
def translate_text(text, target_lang="es"):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text

# Simple Paraphraser (using a mock API as example)
def paraphrase_text(text):
    try:
        api_url = "https://paraphrase-api-example.com/paraphrase"  # Replace with actual API
        data = {'text': text}
        response = requests.post(api_url, json=data)
        result = response.json()  # Mock API response
        return result.get('paraphrased_text', 'No paraphrased text available.')
    except Exception as e:
        return f"Error in paraphrasing: {str(e)}"

# AI-generated text detection (using a mock API)
def detect_ai_generated(text):
    try:
        api_url = "https://ai-detector-api-example.com/detect"  # Replace with actual API
        data = {'text': text}
        response = requests.post(api_url, json=data)
        result = response.json()  # Mock API response
        return result.get('ai_detected', False), result.get('confidence_score', 0)
    except Exception as e:
        return False, f"Error in AI detection: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

# Summarization route
@app.route('/text-summarization', methods=["POST"])
def summarize():
    summary = ""
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        if inputtext:
            input_text = "summarize: " + inputtext
            try:
                tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
                summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
                summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
            except Exception as e:
                summary = f"Error in summarization: {str(e)}"
    
    return render_template("output.html", data={"summary": summary})

# Grammar check route
@app.route('/grammar_checker', methods=["POST"])
def grammar_checker():
    corrected_text = ""
    corrections = []
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        if inputtext:
            try:
                corrected_text, corrections = grammar_check(inputtext)
            except Exception as e:
                corrected_text = f"Error in grammar check: {str(e)}"
    
    return render_template("output.html", data={"corrected_text": corrected_text, "corrections": corrections})

# Plagiarism checker route
@app.route('/plagiarism_checker', methods=["POST"])
def plagiarism_checker():
    plagiarism_percentage = 0
    plagiarism_report = ""
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        if inputtext:
            try:
                plagiarism_percentage, plagiarism_report = check_plagiarism(inputtext)
            except Exception as e:
                plagiarism_report = f"Error in plagiarism check: {str(e)}"
    
    return render_template("output.html", data={"plagiarism_percentage": plagiarism_percentage, "plagiarism_report": plagiarism_report})

# Translation route
@app.route('/translate', methods=["POST"])
def translate():
    translated_text = ""
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        target_lang = request.form.get("target_lang", "es")  # Default to Spanish
        if inputtext:
            try:
                translated_text = translate_text(inputtext, target_lang)
            except Exception as e:
                translated_text = f"Error in translation: {str(e)}"
    
    return render_template("output.html", data={"translated_text": translated_text})

# Paraphraser route
@app.route('/paraphrase', methods=["POST"])
def paraphrase():
    paraphrased_text = ""
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        if inputtext:
            try:
                paraphrased_text = paraphrase_text(inputtext)
            except Exception as e:
                paraphrased_text = f"Error in paraphrasing: {str(e)}"
    
    return render_template("output.html", data={"paraphrased_text": paraphrased_text})

# AI Detector route
@app.route('/ai_detector', methods=["POST"])
def ai_detector():
    ai_detected = False
    confidence_score = 0
    if request.method == "POST":
        inputtext = request.form.get("inputtext_", "")
        if inputtext:
            try:
                ai_detected, confidence_score = detect_ai_generated(inputtext)
            except Exception as e:
                ai_detected = f"Error in AI detection: {str(e)}"
    
    return render_template("output.html", data={"ai_detected": ai_detected, "confidence_score": confidence_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))  # Ensure it runs on Heroku

