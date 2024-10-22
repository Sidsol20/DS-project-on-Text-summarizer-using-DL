from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import language_tool_python
from googletrans import Translator

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

# Translator using googletrans
def translate_text(text, target_lang="es"):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text

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

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()
