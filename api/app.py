from flask import Flask, jsonify
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
import clean
import design
import hypothesis
import pandas as pd

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

@app.route('/api/data_cleaning')
def hello_world():
    df = pd.read_csv("customers-100.csv") # Replace with correct csv loading method
    cleaning_result = clean.data_clean(df, llm)
    return cleaning_result

@app.route("/api/design-procedure")
def design_procedure():
    df = pd.read_csv("customers-100.csv")
    procedure = design.design_procedure(df)
    return procedure

@app.route("/api/hypothesis_visuals")
def hypothesis_visuals():
    df = pd.read_csv("customers-100.csv")
    hypothesis = hypothesis.run_hypothesis_pipeline(df, design_procedure(), llm)
    return jsonify({
        "status": 200,
        "message": hypothesis
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)