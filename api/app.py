from flask import Flask, jsonify
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
import clean
import design
import hypothesis
import pandas as pd
from flask import request

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

@app.route('/api/test')
def test():
    return "Hello, World!"

@app.route('/api/data_cleaning', methods=['POST'])
def clean_data():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    try:
        df = pd.read_csv(file)
        cleaning_result = clean.data_clean(df, llm)
        return cleaning_result
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

@app.route('/api/design_procedure', methods=['POST'])
def design_procedure():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    try:
        df = pd.read_csv(file)
        procedure = design.design_procedure(df)
        return procedure
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

@app.route("/api/hypothesis_visuals", methods=['POST'])
def hypothesis_visuals():
    if 'file' not in request.files:
        return "No file part in the request", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    try:
        df = pd.read_csv(file)
        procedure = design.design_procedure(df)
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500
    return jsonify(hypothesis.hypothesis_testing(df, llm, procedure))

if __name__ == '__main__':
    app.run(debug=True, port=5000)