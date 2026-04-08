from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=message
    )

    return jsonify({"reply": response.output_text})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
