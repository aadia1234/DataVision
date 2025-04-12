import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import utils
import pandas as pd
import numpy as np
from flask import jsonify
import json

clean_data_prompt_template = PromptTemplate.from_template("""
You are an expert data scientist. Your task is to observe an overview of what is contained
in a pandas dataframe and then perform data cleaning on it if it is needed. Observe this overview 
of the dataframe and determine if it needs any cleaning. If it does not, reply with exactly 
NO CLEANING NECESSARY with no additional spacing or text. If it does require cleaning, you 
will need to output a JSON object containing a text string summary of the cleaning you are 
performing and a a python function called clean_dataframe that takes the dataframe as input 
and returns the cleaned dataframe. 

Here is an example of what the JSON object should look like:
                                                          
{{
    "summary": "Removed duplicates and filled missing values",
    "clean_dataframe": "def clean_dataframe(df):\n    df = df.drop_duplicates()\n    df = df.fillna(0)\n    return df"
}}

Respond with only the JSON object and no additional 
text or explanation.

Here is the overview of the dataframe:
{data_overview}
""")

def data_clean(df, llm):
    data_overview = utils.overview_data(df)
    clean_data_prompt = clean_data_prompt_template.format(data_overview=data_overview)
    llm_output = utils.cleanCode(llm.invoke(clean_data_prompt))
    if llm_output == "NO CLEANING NECESSARY":
        return jsonify({"summary": "NO CLEANING NECESSARY", "code": None, "cleaned_df": None})
    else:
        json_output = json.loads(llm_output)
        summary = json_output['summary']
        code = json_output['clean_dataframe']
        try:
            local_env = {}
            exec(code, globals(), local_env)
            clean_function = local_env.get("clean_dataframe")

            if clean_function:
                cleaned_df = clean_function(df)
                return jsonify({
                    "summary": summary,
                    "code": code,
                    "cleaned_df": cleaned_df.to_dict(orient='records')
                })
            else:
                return jsonify({"error": "Cleaning function not found in generated code."})
        except Exception as e:
            return jsonify({"error": "Error running generated cleaning code: " + str(e)})