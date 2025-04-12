import ast
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import utils

def run_hypothesis_pipeline(df, input_features, llm):
    def plot_to_base64():
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def get_llm_response(prompt):
        response = llm.invoke(prompt)
        return utils.cleanCode(
            response if isinstance(response, str)
            else getattr(response, "content", "") or response.get("content", "")
        )

    code_prompt_template = PromptTemplate.from_template("""
    Generate Python code for hypothesis testing between features in a dataframe that:
    1. Performs ALL specified tests in the analysis descriptions
    2. Creates high-quality visualizations for ALL relationships
    3. Shows test statistics on plots
    4. Handles different data types appropriately
    5. Returns a list of base64 encoded strings for each plot generated

    IMPORTANT: Instead of using plt.show(), append each plot to a list as a base64 encoded string.

    Analysis descriptions:
    {features}

    Dataframe overview:
    {overview}

    Return ONLY executable code that includes a function to perform all tests and return the list of plot images.
    """)

    initial_prompt = code_prompt_template.format(
        features=input_features,
        overview=utils.overview_data(df)
    )

    code_candidate = get_llm_response(initial_prompt)

    try:
        ast.parse(code_candidate)
    except SyntaxError as syn_err:
        raise ValueError(f"Syntax error in generated code: {syn_err}\n\nCode:\n{code_candidate}")

    exec_globals = {
        '__builtins__': __builtins__,
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'stats': stats,
        'np': np,
        'base64': base64,
        'BytesIO': BytesIO,
        'plot_to_base64': plot_to_base64
    }

    plot_images = []
    try:
        exec(code_candidate, exec_globals)
        plot_images = exec_globals.get('plot_images', [])
    except Exception as e:
        print(f"Error in execution: {e}")
        
    return plot_images

