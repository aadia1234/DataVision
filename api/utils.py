import io 
import base64
import os
from flask import send_file, request, jsonify
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful data analysis assistant. Keep responses concise and focused on the current task."""

def cleanCode(code):
    return code.replace("```python", "").replace("```json", "").replace("```", "")

def overview_data(df):
    output = "Data Overview:\n"
    output += "Columns:\n" + ", ".join(df.columns) + "\n"
    output += "Head:\n" + df.head().to_string() + "\n"
    output += "Tail:\n" + df.tail().to_string() + "\n"
    output += "Sample:\n" + df.sample(5).to_string() + "\n"
    output += "Info:\n"
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    buffer.seek(0)
    output += buffer.read() + "\n"
    output += "Describe:\n" + df.describe().to_string() + "\n"
    return output

def convert_plt_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def send_figure_as_response(fig, fmt='png', dpi=100, bbox_inches='tight', **savefig_kwargs):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    buf.seek(0)
    return send_file(buf, mimetype=f"image/{fmt}")

def chatllm(data, llm):
    """Handles chat with conversation history"""
    try:
        user_msg = data.get("message", "")
        history = data.get("history", [])

        conversation = history + [{"role": "user", "content": user_msg}]
        response = llm.invoke(conversation)

        reply = response.get("content", "")

        return jsonify({
            "reply": reply,
            "history": conversation + [{"role": "assistant", "content": reply}]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500