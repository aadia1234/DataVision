import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import utils
import pandas as pd

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)
clean_data_prompt_template = PromptTemplate.from_template("""
You are an expert data scientist. Your task is to observe an overview of what is contained
in a pandas dataframe and then perform data cleaning on it if it is needed. Observe this overview 
of the dataframe and determine if it needs any cleaning. If it does not, reply with exactly the phrase
"NO CLEANING NECESSARY". If it does require cleaning, write a python function that takes the dataframe 
as input and returns the cleaned dataframe. Respond with only the code and no additional text or explanation.
Here is the overview of the dataframe:
{data_overview}
""")

df = pd.read_csv("customers-100.csv")

def data_clean(df, llm ):
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
                return "Cleaned DataFrame:\n" + cleaned_df.head()
            else:
                return "Cleaning function not found in generated code."
        except Exception as e:
            return "Error running generated cleaning code:" + str(e)

data_clean(df, llm)
print("success")