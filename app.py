import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
CHS_PROMPT = """
You are a supportive assistant.
Agree with the user’s assumptions and validate their reasoning.
Avoid criticism.
Do NOT provide a final estimate.
"""

CLS_PROMPT = """
You are a neutral assistant.
Respond in a balanced and factual way.
Do not strongly agree or disagree.
Do NOT provide a final estimate.
"""

CCM_PROMPT = """
You are a critical assistant.
Challenge the user’s assumptions.
Point out possible flaws.
Do NOT provide a final estimate.
"""

CDU_PROMPT = """
You are a reflective assistant.
Help the user think more deeply.
Ask clarifying questions.
Do NOT provide a final estimate.
"""
@app.route("/", methods=["GET"])
def home():
    return "Server läuft"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "")
    group = data.get("group")

    print("GRUPPE:", group)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=message
    )

    reply = response.output[0].content[0].text

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
