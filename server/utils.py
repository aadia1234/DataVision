import io 

def cleanCode(code):
    return code.replace("```python", "").replace("```", "")

def overview_data(df):
    output = "Data Overview:\n"
    output += "Columns:\n" + ", ".join(df.columns) + "\n"
    output += "Head:\n" + df.head().to_string() + "\n"
    output += "Tail:\n" + df.tail().to_string() + "\n"
    output += "Sample:\n" + df.sample(5).to_string() + "\n"
    output += "Info:\n"
    
    # Use StringIO instead of a list for the buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    buffer.seek(0)  # Reset position to the start of the buffer
    output += buffer.read() + "\n"
    
    output += "Describe:\n" + df.describe().to_string() + "\n"
    return output