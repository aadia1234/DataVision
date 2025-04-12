import io 
import base64
from flask import send_file

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

def convert_plt_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def send_figure_as_response(fig, fmt='png', dpi=100, bbox_inches='tight', **savefig_kwargs):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    buf.seek(0)
    mimetype = f"image/{fmt}"
    return send_file(buf, mimetype=mimetype)