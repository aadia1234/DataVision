import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import re
# Helper functions for data preprocessing
def safe_convert_to_datetime(series):
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception as e:
        print(f'Error converting to datetime: {e}')
        return series

# df is assumed to be defined


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, kruskal, f_oneway

def perform_hypothesis_tests(df):
    """
    Performs hypothesis tests and generates visualizations for relationships between features in a dataframe.
    """

    def chi_squared_test(df, col1, col2):
        """Performs Chi-squared test (with Fisher's exact test fallback) and visualizes the relationship."""
        # Handle missing values
        df_clean = df[[col1, col2]].dropna()

        # Ensure that the cleaned data has at least two unique values in each column
        if df_clean[col1].nunique() < 2 or df_clean[col2].nunique() < 2:
            print(f"Skipping Chi-squared test for {col1} and {col2}: Insufficient unique values.")
            return np.nan

        contingency_table = pd.crosstab(df_clean[col1], df_clean[col2])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        min_expected = np.min(
            np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / np.sum(contingency_table)
        )

        if min_expected < 5:
            try:
                oddsratio, p = fisher_exact(contingency_table)
                test_name = "Fisher's Exact Test"
                test_result = f"Odds Ratio: {oddsratio:.3f}, p-value: {p:.3e}"

            except ValueError:
                test_name = "Chi-squared Test (insufficient data for Fisher's exact)"
                test_result = "Cannot perform test due to contingency table structure"
                p = np.nan
        else:
            test_name = "Chi-squared Test"
            test_result = f"Chi2: {chi2:.3f}, p-value: {p:.3e}"

        # Plotting
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=col1, hue=col2, data=df_clean)
        plt.title(f"Relationship between {col1} and {col2}", fontsize=16, fontweight="bold")
        plt.xlabel(col1, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=False)

        # Add statistical results to the plot
        plt.text(0.05, 0.95, test_name, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        plt.text(0.05, 0.90, test_result, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        plt.show()

        return p


    def anova_test(df, col1, col2):
        """Performs ANOVA (or Kruskal-Wallis if needed) and visualizes the relationship."""
        df_clean = df[[col1, col2]].dropna()

        # Check if the date column has sufficient unique values.
        if df_clean[col2].nunique() < 2:
            print(f"Skipping ANOVA/Kruskal-Wallis test for {col1} and {col2}: Insufficient unique values.")
            return np.nan

        # Check if the date column is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean[col1]):
            try:
                dates = pd.to_datetime(df_clean[col1], errors='coerce')
            except ValueError as e:
                print(f"Error converting {col1} to datetime: {e}")
                return np.nan

        else:
            dates = df_clean[col1]

        dates = dates.dropna()

        if len(dates) == 0:
            print("No valid dates to perform test.")
            return np.nan

        # Group the numeric dates by the categorical variable
        groups = df_clean[col2].dropna().unique()
        grouped_data = []

        for g in groups:
            group_dates = dates[df_clean[col2].dropna() == g]
            grouped_data.append(group_dates)

        # Check for sufficient sample sizes in each group
        group_sizes = [len(g) for g in grouped_data]
        min_group_size = min(group_sizes)

        if min_group_size < 2 or len(groups) < 2:
            stat, p = kruskal(*[g.astype(np.int64) for g in grouped_data])
            test_name = "Kruskal-Wallis Test (small sample size or insufficient groups)"
            test_result = f"Statistic: {stat:.3f}, p-value: {p:.3e}"

        else:
            # Perform ANOVA if possible, otherwise Kruskal-Wallis
            try:
                fvalue, pvalue = f_oneway(*[g.astype(np.int64) for g in grouped_data])
                test_name = "ANOVA"
                test_result = f"F: {fvalue:.3f}, p-value: {pvalue:.3e}"
                p = pvalue
            except Exception:
                stat, p = kruskal(*[g.astype(np.int64) for g in grouped_data])
                test_name = "Kruskal-Wallis Test (ANOVA failed)"
                test_result = f"Statistic: {stat:.3f}, p-value: {p:.3e}"


        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col2, y=dates.astype(np.int64), data=df_clean)
        plt.title(f"Relationship between {col1} and {col2}", fontsize=16, fontweight="bold")
        plt.xlabel(col2, fontsize=14)
        plt.ylabel(col1, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        plt.text(0.05, 0.95, test_name, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        plt.text(0.05, 0.90, test_result, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        plt.show()

        return p

    # Load the dataframe (replace with your actual data loading)
    data = {'Subscription Date': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-01', '2023-02-25', None],
            'Country': ['USA', 'Canada', 'USA', 'UK', 'Canada', 'USA'],
            'Company Type': ['Tech', 'Finance', 'Tech', 'Retail', 'Finance', 'Tech'],
            'Email': ['john.doe@gmail.com', 'jane.smith@yahoo.ca', 'peter.jones@company.com', 'lisa.brown@retail.co.uk', 'mark.wilson@finance.ca', None],
            'Website': ['www.techco.com', 'www.financeinc.ca', 'www.techsolutions.com', 'www.retailgroup.co.uk', 'www.finance.ca', None],
            'Phone Number': ['123-456-7890', '456-789-0123', '789-012-3456', '0123456789', '345-678-9012', None]}
    df = pd.DataFrame(data)

    # Feature Engineering
    df['Email Domain'] = df['Email'].str.split('@').str[1].str.split('.').str[0]
    df['Website Domain'] = df['Website'].str.replace('www.', '', regex=False).str.split('.').str[0]
    df['Phone Format'] = df['Phone Number'].str.replace(r'\d', '', regex=True)

    # Analysis descriptions
    tests = [
        ("Country", "Company Type", "Chi-squared"),
        ("Country", "Subscription Date", "ANOVA"),
        ("Email Domain", "Country", "Chi-squared"),
        ("Website Domain", "Company Type", "Chi-squared"),
        ("Phone Format", "Country", "Chi-squared"),
    ]

    # Perform tests based on descriptions
    for col1, col2, test_type in tests:
        print(f"Performing {test_type} test for {col1} and {col2}")
        if test_type == "Chi-squared":
            chi_squared_test(df, col1, col2)
        elif test_type == "ANOVA":
            anova_test(df, col1, col2)


perform_hypothesis_tests(pd.DataFrame())


# Automatically run the function
if __name__ == '__main__':
    # Extract the function name from the code
    import re
    function_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(\s*df\s*\)', __file__)
    if function_match:
        function_name = function_match.group(1)
        # Call the function with the dataframe
        globals()[function_name](df)
    else:
        print('Could not find the main function to call')
