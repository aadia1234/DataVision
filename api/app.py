from flask import Flask, jsonify, request
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
import clean
import design
import hypothesis
import summarize
import utils
import pandas as pd
import uuid

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_file_path(id):
    return os.path.join(UPLOAD_FOLDER, f"{id}.csv")


@app.route('/api/test')
def test():
    return "Hello, World!"

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    try:
        id = str(uuid.uuid4())
        filepath = get_file_path(id)
        file.save(filepath)
        return jsonify({"id": id})
    except Exception as e:
        return f"An error occurred while saving the file: {str(e)}", 500

@app.route('/api/data_cleaning', methods=['POST'])
def clean_data():
    id = request.json.get('id')
        
    try:
        df = pd.read_csv(get_file_path(id))
        cleaning_result = clean.data_clean(df, llm)
        return cleaning_result
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

@app.route('/api/design_procedure', methods=['POST'])
def design_procedure():
    id = request.json.get('id')
    
    try:
        df = pd.read_csv(get_file_path(id))
        procedure = design.design_procedure(df)
        return procedure
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

@app.route("/api/hypothesis_test", methods=['POST'])
def hypothesis_test():
    id = request.json.get('id')

    try:
        df = pd.read_csv(get_file_path(id))
        procedure = request.json.get('designResult')
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500
    return jsonify(hypothesis.hypothesis_testing(df, llm, procedure))

@app.route("/api/summarize", methods=['POST'])
def summarize_analysis():
    data = request.get_json()
    cleaning_summary = data.get('cleaning_summary')
    potential_relationships = data.get('potential_relationships')
    p_values_summary = data.get('p_values_summary')
    return summarize.summarize(cleaning_summary, potential_relationships, p_values_summary, llm)


@app.route("/api/chat", methods=['POST'])
def chat():
    data = request.get_json()
    return utils.chatllm(data, llm)

if __name__ == '__main__':
    app.run(debug=True, port=5000)