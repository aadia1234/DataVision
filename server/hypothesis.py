import os
import ast
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import utils
import matplotlib
matplotlib.use('TkAgg')

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

<<<<<<< HEAD
prompt_template = PromptTemplate.from_template("""
Generate python code for hypothesis testing for the following features using matplotlib, pandas, scipy, seaborn. 
Be sure to generate high quality plots. If the plots are not sufficient, please output further instructions to improve them. 
When the plots are optimized to the best quality, include the comment "#PLOT_FINAL" at the end of your code.
Input: {input}
=======
code_ht_prompt_template = PromptTemplate.from_template("""

    You are an expert data scientist. Your task is to generate python code for hypothesis 
    testing for relationships between certain features in a dataframe. The code should run the test,
    calculating the p value, test statistic, and display the results in an appropriate graph. You 
    will be provided with the following things: 
        1. a list of features with which hypothesis tests to conduct for each features
        2. an overview of the dataframe you are working with
    
    Your output should be a python function that performs this hypothesis testing and visualization.
    You should use matplotlib, scipy, pandas, and seaborn for this.
    Return only the code and no additional text. 
    
    Here is the list of features: 
    {features}
    
    Here is an overview of the dataframe:
    {overview}
>>>>>>> main
""")

def run_hypothesis_analysis(df, input_features, llm):
    try:
<<<<<<< HEAD
        max_iterations = 5
        final_code = None
        previous_code = None
        iteration = 0
=======
        # Format the prompt
        prompt = code_ht_prompt_template.format(features=input_features, overview=utils.overview_data(df))
>>>>>>> main

        while iteration < max_iterations:
            if previous_code is None:
                prompt = prompt_template.format(input=input_features)
            else:
                prompt = (
                    "Improve the following python code for hypothesis testing to ensure that each plot is of the highest possible quality "
                    "using matplotlib, pandas, scipy, and seaborn. Retain the original functionality. If you believe the plots are now optimal, "
                    "include the comment '#PLOT_FINAL' at the end of your code. Here is your previous code:\n\n" + previous_code
                )

            response = llm.invoke(prompt)
            code_candidate = utils.cleanCode(response)

            # Check if the code ends with the #PLOT_FINAL marker
            if "#PLOT_FINAL" in code_candidate:
                final_code = code_candidate
                break
            else:
                previous_code = code_candidate

            iteration += 1

        # Fallback to last generated code if marker not found
        if final_code is None:
            final_code = previous_code

        # Optional syntax check
        try:
            ast.parse(final_code)
        except SyntaxError as syn_err:
            return None, f"Syntax error in generated code: {syn_err}\n\nGenerated Code:\n{final_code}"

        # Write code safely to file
        with open('generated_tests.py', 'w') as f:
            f.write("# Required imports\n")
            f.write("import pandas as pd\n")
            f.write("import matplotlib.pyplot as plt\n")
            f.write("import seaborn as sns\n")
            f.write("from scipy import stats\n")
            f.write("import numpy as np\n\n")
            f.write("# df is assumed to be defined\n\n")
            f.write("# Generated hypothesis testing code starts here\n")
            f.write(final_code)
            f.write("\n")

        # Execute code with df context
        exec_globals = {
            '__builtins__': __builtins__,
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'np': np
        }

        with open('generated_tests.py') as f:
            exec(f.read(), exec_globals)

        plt.show()
        return final_code, "Hypothesis tests completed successfully"

    except Exception as e:
        return None, f"Error running hypothesis tests: {e}"


# Run the hypothesis testing
if __name__ == "__main__":
    input_features = """
    Subscription Date vs Country
    Company Type vs Country
    Email Domain vs Country
    Website Domain vs Company Type
    Phone Format vs Country
    """

    df = pd.DataFrame({
        'Subscription Date': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-01', '2023-02-25'],
        'Country': ['USA', 'Canada', 'USA', 'UK', 'Canada'],
        'Company Type': ['Tech', 'Finance', 'Tech', 'Retail', 'Finance'],
        'Email': ['john.doe@gmail.com', 'jane.smith@yahoo.ca', 'peter.jones@company.com', 'lisa.brown@retail.co.uk', 'mark.wilson@finance.ca'],
        'Website': ['www.techco.com', 'www.financeinc.ca', 'www.techsolutions.com', 'www.retailgroup.co.uk', 'www.finance.ca'],
        'Phone Number': ['123-456-7890', '456-789-0123', '789-012-3456', '0123456789', '345-678-9012']
    })

    code, result = run_hypothesis_analysis(df, input_features, llm)

    if code:
        print("Generated Hypothesis Testing Code:")
        print(code)
        print("\nTest Execution Result:")
        print(result)
    else:
        print("Error encountered during hypothesis testing:")
        print(result)
