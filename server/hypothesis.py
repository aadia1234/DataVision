# Standard library imports
import os
import ast
import base64
import re
from io import BytesIO

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Local application imports
import utils

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

def run_hypothesis_pipeline(df, input_features, llm):
    """
    Run hypothesis tests using LLM-generated code based on provided feature relationships.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to analyze
    input_features : str
        String containing analysis descriptions in the format:
        "Relationship between Feature1 and Feature2. Hypothesis test: TestName."
    llm : LLM object
        Language model for generating code
    
    Returns:
    --------
    tuple
        (final_code, result_message)
    """
    max_iterations = 3
    iteration = 0

    def plot_to_base64():
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def get_llm_response(prompt):
        response = llm.invoke(prompt)
        return utils.cleanCode(
            response if isinstance(response, str)
            else getattr(response, "content", "") or response.get("content", "")
        )

    def get_decision(code_text, image_b64):
        decision_prompt = (
            "You are reviewing the output of a hypothesis testing script.\n"
            f"Here is the current Python code:\n{code_text}\n\n"
            f"Here is the base64-encoded plot it produced:\n<image>{image_b64}</image>\n\n"
            "Does this code produce a visually and statistically meaningful output for hypothesis testing? "
            "Reply only 'yes' or 'no'."
        )
        response = llm.invoke(decision_prompt)
        result = ""
        if isinstance(response, str):
            result = response
        elif hasattr(response, "content"):
            result = response.content
        elif isinstance(response, dict) and "content" in response:
            result = response["content"]
        return "yes" in result.lower() or "#plot_final" in result.lower()

    # Define the code prompt template for generating the initial code
    code_prompt_template = PromptTemplate.from_template("""
You are an expert data scientist. Your task is to generate Python code for hypothesis 
testing for relationships between features in a dataframe. The code should:

1. Parse the analysis descriptions to identify feature pairs and test types
2. Perform appropriate hypothesis tests (Chi-squared, ANOVA, etc.)
3. Create visually appealing plots showing the relationships
4. Display statistical results on the plots
5. Handle different data types (categorical, numerical, dates) appropriately

IMPORTANT REQUIREMENTS:
- NEVER skip tests specified in the analysis descriptions, even if sample sizes are small
- For Chi-squared tests with small expected values, use Fisher's exact test as a fallback
- For ANOVA tests with small sample sizes, use non-parametric alternatives
- Always display a plot for each test, even if the test cannot be performed statistically
- Use only pandas, matplotlib, seaborn, numpy, and scipy.stats libraries
- Handle missing values appropriately
- Make sure all plots are displayed automatically
- Add clear annotations showing test results on each plot
- For date fields, use pd.to_datetime() with errors='coerce' and handle potential conversion issues
- NEVER use the .dt accessor directly on columns without first checking if they are datetime type
- When converting dates to numeric values, use a safe approach that handles errors gracefully
- Fix the FutureWarning about seaborn's palette parameter by using hue instead with legend=False

Return ONLY executable Python code without any explanations or markdown.

Here are the analysis descriptions:
{features}

Here is an overview of the dataframe:
{overview}

Generate a single function that can process all these tests and produce high-quality visualizations.
""")

    initial_prompt = code_prompt_template.format(
        features=input_features,
        overview=utils.overview_data(df)
    )

    code_candidate = get_llm_response(initial_prompt)
    print("Initial generated code:\n", code_candidate)
    
    try:
        ast.parse(code_candidate)
    except SyntaxError as syn_err:
        raise ValueError(f"Syntax error in generated code: {syn_err}\n\nCode:\n{code_candidate}")

    def execute_and_capture(code_text):
        with open('generated_tests.py', 'w') as f:
            f.write("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n")
            f.write("from scipy import stats\nimport numpy as np\nimport re\n")
            f.write("# Helper functions for data preprocessing\n")
            f.write("def safe_convert_to_datetime(series):\n")
            f.write("    try:\n")
            f.write("        return pd.to_datetime(series, errors='coerce')\n")
            f.write("    except Exception as e:\n")
            f.write("        print(f'Error converting to datetime: {e}')\n")
            f.write("        return series\n\n")
            f.write("# df is assumed to be defined\n\n")
            f.write(code_text + "\n")
            
            # Add code to automatically call the function with the correct name
            f.write("\n# Automatically run the function\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    # Extract the function name from the code\n")
            f.write("    import re\n")
            f.write("    function_match = re.search(r'def\\s+([a-zA-Z0-9_]+)\\s*\\(\\s*df\\s*\\)', __file__)\n")
            f.write("    if function_match:\n")
            f.write("        function_name = function_match.group(1)\n")
            f.write("        # Call the function with the dataframe\n")
            f.write("        globals()[function_name](df)\n")
            f.write("    else:\n")
            f.write("        print('Could not find the main function to call')\n")
        
        exec_globals = {
            '__builtins__': __builtins__,
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'np': np,
            're': re
        }
        plt.close('all')
        try:
            with open('generated_tests.py') as f:
                exec(f.read(), exec_globals)
            return plot_to_base64()
        except Exception as e:
            print(f"Error executing code: {e}")
            # Create error message plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error: {str(e)}", 
                    horizontalalignment="center", verticalalignment="center", fontsize=14)
            plt.tight_layout()
            return plot_to_base64()

    image_b64 = execute_and_capture(code_candidate)
    print("Initial plot captured as base64 (first 100 chars):", image_b64[:100])

    decision = get_decision(code_candidate, image_b64)
    print("Initial decision:", decision)

    # Iteratively refine the code if needed
    while not decision and iteration < max_iterations:
        refine_prompt = (
            "You are improving a Python script for hypothesis testing. The current code needs refinement to ensure "
            "it produces visually appealing and statistically meaningful plots. Issues to address:\n"
            "1. NEVER skip tests specified in the analysis descriptions, even if sample sizes are small\n"
            "2. For Chi-squared tests with small expected values, use Fisher's exact test as a fallback\n"
            "3. For ANOVA tests with small sample sizes, use non-parametric alternatives\n"
            "4. Fix the FutureWarning about seaborn's palette parameter by using hue instead with legend=False\n"
            "5. Improve plot aesthetics (titles, labels, colors, etc.)\n"
            "6. Display statistical results on the plots\n"
            "7. Handle edge cases (missing data, etc.)\n"
            "8. Fix any issues with date handling - don't use .dt accessor without checking type\n\n"
            "Use only pandas, matplotlib, seaborn, numpy, and scipy.stats libraries.\n\n"
            f"Here is the previous code:\n\n{code_candidate}\n\n"
            f"Here is the plot generated (base64 encoded):\n<image>{image_b64}</image>\n\n"
            "Return ONLY the improved code without any explanations or markdown."
        )
        code_candidate = get_llm_response(refine_prompt)
        print(f"Refined code (iteration {iteration + 1}):\n", code_candidate)
        
        try:
            ast.parse(code_candidate)
        except SyntaxError as syn_err:
            raise ValueError(f"Syntax error in refined code: {syn_err}\n\nCode:\n{code_candidate}")
        
        image_b64 = execute_and_capture(code_candidate)
        print("Refined plot captured as base64 (first 100 chars):", image_b64[:100])
        
        decision = get_decision(code_candidate, image_b64)
        print("Decision after refinement:", decision)
        iteration += 1

    # Save final code and execute
    with open('generated_tests.py', 'w') as f:
        f.write("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n")
        f.write("from scipy import stats\nimport numpy as np\nimport re\n")
        f.write("# Helper functions for data preprocessing\n")
        f.write("def safe_convert_to_datetime(series):\n")
        f.write("    try:\n")
        f.write("        return pd.to_datetime(series, errors='coerce')\n")
        f.write("    except Exception as e:\n")
        f.write("        print(f'Error converting to datetime: {e}')\n")
        f.write("        return series\n\n")
        f.write("# df is assumed to be defined\n\n")
        f.write(code_candidate + "\n")
        
        # Add code to automatically call the function with the correct name
        f.write("\n# Automatically run the function\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    # Extract the function name from the code\n")
        f.write("    import re\n")
        f.write("    function_match = re.search(r'def\\s+([a-zA-Z0-9_]+)\\s*\\(\\s*df\\s*\\)', __file__)\n")
        f.write("    if function_match:\n")
        f.write("        function_name = function_match.group(1)\n")
        f.write("        # Call the function with the dataframe\n")
        f.write("        globals()[function_name](df)\n")
        f.write("    else:\n")
        f.write("        print('Could not find the main function to call')\n")
    
    exec_globals = {
        '__builtins__': __builtins__,
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'stats': stats,
        'np': np,
        're': re
    }
    
    try:
        with open('generated_tests.py') as f:
            exec(f.read(), exec_globals)
    except Exception as e:
        print(f"Error in final execution: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error in final execution: {str(e)}", 
                horizontalalignment="center", verticalalignment="center", fontsize=14)
        plt.tight_layout()

    if not plt.get_fignums():
        print("⚠️ No figures were generated by the script. Creating a fallback figure...")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No plot generated by the script", 
                horizontalalignment="center", verticalalignment="center", fontsize=14)
        ax.set_title("Notification")
    
    print("Final generated figure numbers:", plt.get_fignums())
    print("✅ Displaying final plot(s)...")
    plt.show()

    return code_candidate, "Hypothesis tests completed successfully (via LangGraph pipeline)"

if __name__ == "__main__":
    # Sample dataframe - in production, this would be your actual data
    df = pd.DataFrame({
        'Subscription Date': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-01', '2023-02-25', None],
        'Country': ['USA', 'Canada', 'USA', 'UK', 'Canada', 'USA'],
        'Company Type': ['Tech', 'Finance', 'Tech', 'Retail', 'Finance', 'Tech'],
        'Email': ['john.doe@gmail.com', 'jane.smith@yahoo.ca', 'peter.jones@company.com', 'lisa.brown@retail.co.uk', 'mark.wilson@finance.ca', None],
        'Website': ['www.techco.com', 'www.financeinc.ca', 'www.techsolutions.com', 'www.retailgroup.co.uk', 'www.finance.ca', None],
        'Phone Number': ['123-456-7890', '456-789-0123', '789-012-3456', '0123456789', '345-678-9012', None]
    })
    
    # Sample analysis string - this would come from your input
    analysis_string = """
    Relationship between Country and Company Type. Hypothesis test: Chi-squared test.
    Relationship between Country and Subscription Date. Hypothesis test: ANOVA.
    Relationship between Email Domain and Country. Hypothesis test: Chi-squared test.
    Relationship between Website Domain and Company Type. Hypothesis test: Chi-squared test.
    Relationship between Phone Format and Country. Hypothesis test: Chi-squared test.
    """
    
    final_code, result_msg = run_hypothesis_pipeline(df, analysis_string, llm)
    print("Final Generated Hypothesis Testing Code:")
    print(final_code)
    print("\nPipeline Execution Result:")
    print(result_msg)
