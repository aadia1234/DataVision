import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import utils
import pandas as pd
import numpy as np
from flask import jsonify

clean_data_prompt_template = PromptTemplate.from_template("""
You are an expert data scientist. Your task is to observe an overview of what is contained
in a pandas dataframe and then perform data cleaning on it if it is needed. Observe this overview 
of the dataframe and determine if it needs any cleaning. If it does not, reply with exactly the phrase
"NO CLEANING NECESSARY". If it does require cleaning, write a python function called clean_dataframe that 
takes the dataframe as input and returns the cleaned dataframe. Respond with only the code and no additional 
text or explanation.

Here is the overview of the dataframe:
{data_overview}
""")

def data_clean(df, llm):
    data_overview = utils.overview_data(df)
    clean_data_prompt = clean_data_prompt_template.format(data_overview=data_overview)
    code = utils.cleanCode(llm.invoke(clean_data_prompt))
    print("CODE: \n", code)
    if code.strip() == "NO CLEANING NECESSARY":
        return "No cleaning was needed."
    else:
        try:
            local_env = {}
            exec(code, globals(), local_env)
            clean_function = local_env.get("clean_dataframe")

            if clean_function:
                cleaned_df = clean_function(df)
                return jsonify(cleaned_df.to_dict(orient='records'))
            else:
                return "Cleaning function not found in generated code."
        except Exception as e:
            return "Error running generated cleaning code:" + str(e)