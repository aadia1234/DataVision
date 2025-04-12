import os
import ast
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import utils

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

df = pd.read_csv("customers-100.csv")

hypothesis_testing_prompt_template = PromptTemplate.from_template("""
    You are an expert data scientist. Your task is to write a function that conducts a series of 
    hypothesis tests on a dataframe, examining the relationships between different columns in the 
    dataframe.
    
    You will be provided with the following items:
    1. A set of instructions containing which hypothesis tests to run and the columns to use.
    2. An overview of the structure of the dataframe that will be inputted to your function.
    
    Write a function called hypothesis_test that takes in a pandas dataframe and performs the specified 
    hypothesis tests using scipy. You can assume that the dataframe will always have the properties detailed in the 
    dataframe overview you receive. 
    
    For each set of features, you should also create a visualization of the data you are examining with matplotlib 
    and seaborn. 
    
    Have the function return an array of all of the figures created.
    
    Only use the following libraries:
    - pandas
    - matplotlib
    - seaborn
    - scipy.stats
    - numpy
    
    Respond with only the code and no additional text or explanation.

    Here are the instructions:
    {instructions}
    
    Here is the overview of the dataframe:
    {data_overview}

""")

def hypothesis_testing(df, llm, instructions):
    data_overview = utils.overview_data(df)
    hypothesis_testing_prompt = hypothesis_testing_prompt_template.format(data_overview=data_overview, instructions=instructions)
    code = utils.cleanCode(llm.invoke(hypothesis_testing_prompt))
    print("CODE: \n", code)
    plotstrings = []

    try:
        local_env = {}
        exec(code, local_env)
        hypothesis_test = local_env.get("hypothesis_test")

        if hypothesis_test:
            figures = hypothesis_test(df)
            figures[0].savefig("firstgraph.png")
            figures = hypothesis_test(df)
            base64_figures = [utils.convert_plt_to_base64(fig) for fig in figures]
            return {
                "status": "success",
                "figures": base64_figures
            }
        else:
            return "Hypothesis testing function not found in generated code."
    except Exception as e:
        return "Error running generated hypothesis testing code:" + str(e)



test_procedure = """

Analysis Results:
Relationship between Country and Company. Hypothesis test: Chi-squared test.
Relationship between Country and Subscription Date. Hypothesis test: ANOVA.
Relationship between City and Company. Hypothesis test: Chi-squared test.
Relationship between Subscription Date and Company. Hypothesis test: ANOVA.

"""

print(hypothesis_testing(df, llm, test_procedure))