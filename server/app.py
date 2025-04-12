from flask import Flask
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
import clean
import pandas as pd

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

@app.route('/api/data_cleaning')
def hello_world():
    df = pd.read_csv("customers-100.csv") # Replace with correct csv loading method
    cleaning_result = clean.data_clean(df, llm)
    return cleaning_result

if __name__ == '__main__':
    app.run(debug=True, port=5000)