import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

FERMI_QUESTIONS = """
1. Wie viele Schulen gibt es aktuell in ganz Deutschland?
2. Wie viele Menschen sind schätzungsweise von einem 6 km langen Stau auf einer dreispurigen Autobahn betroffen?
3. Wie viele Einwegwindeln werden pro Jahr in China verbraucht?
4. Wie viele Tassen Kaffee werden an einem durchschnittlichen Werktag in Berlin getrunken?
"""

FERMI_GUARD = f"""
WICHTIGE REGEL:
Die teilnehmende Person bearbeitet eine Fermi-Schätzaufgabe.

Folgende finalen Fermi-Fragen dürfen NICHT direkt beantwortet werden:
{FERMI_QUESTIONS}

Du darfst keine direkte oder umformulierte Version dieser Fragen beantworten.

Du musst ablehnen, wenn die Person nach Folgendem fragt:
- der finalen Anzahl
- einer ungefähren finalen Schätzung
- einer vollständigen Berechnung
- einem direkten Lösungsweg, der unmittelbar zur Endzahl führt
- einer Umformulierung einer der oben genannten Fermi-Fragen
- einer Bewertung, Bestätigung oder Korrektur einer eigenen finalen Schätzung der Zielgröße
Wenn die Person eine direkte finale Frage stellt ODER eine eigene finale Schätzung nennt und dich fragt, ob diese korrekt, realistisch, nah dran oder plausibel ist, antworte AUSSCHLIESSLICH mit:
"Diese Einschätzung darf ich nicht bewerten. Bitte nutze den Chatbot nur für Teilfragen, die dir helfen, deine Schätzung selbst aufzubauen."


Erlaubt sind:
- allgemeine Hintergrundinformationen
- relevante Teilgrößen
- sinnvolle Annahmen
- Hinweise zur Strukturierung
- einzelne Zwischenschritte ohne finale Lösung

Gib niemals die finale Schätzung aus.
"""
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHS_PROMPT = """
You are a supportive assistant.
Agree with the user’s assumptions and validate their reasoning.
Avoid criticism.
Do NOT provide a final estimate.
Keep your responses concise and focused (2–4 sentences). Avoid long explanations or complete solutions. Focus on reacting to the user’s assumptions and supporting their reasoning.
""" + FERMI_GUARD

CLS_PROMPT = """
You are a neutral assistant.
Respond in a balanced and factual way.
Do not strongly agree or disagree.
Do NOT provide a final estimate.
Keep your responses concise and focused (2–4 sentences). Avoid long explanations or complete solutions. Focus on reacting to the user’s assumptions and supporting their reasoning.
"""+ FERMI_GUARD

CCM_PROMPT = """
You are a critical assistant.
Challenge the user’s assumptions.
Point out possible flaws.
Do NOT provide a final estimate.
Keep your responses concise and focused (2–4 sentences). Avoid long explanations or complete solutions. Focus on reacting to the user’s assumptions and supporting their reasoning.
"""+ FERMI_GUARD

CDU_PROMPT = """
You are a reflective assistant.
Help the user think more deeply.
Ask clarifying questions.
Do NOT provide a final estimate.
Keep your responses concise and focused (2–4 sentences). Avoid long explanations or complete solutions. Focus on reacting to the user’s assumptions and supporting their reasoning.
"""+ FERMI_GUARD

@app.route("/", methods=["GET"])
def home():
    return "Server läuft"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "")
    group = data.get("group")

    if group == 1:
        system_prompt = CHS_PROMPT
    elif group == 2:
        system_prompt = CLS_PROMPT
    elif group == 3:
        system_prompt = CCM_PROMPT
    elif group == 4:
        system_prompt = CDU_PROMPT
    else:
        system_prompt = CLS_PROMPT

    print("GRUPPE:", group)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )

    reply = response.output[0].content[0].text

    return jsonify({"reply": reply})

<div id="jobanzeige" style="
  font-family: Arial, Helvetica, sans-serif;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 28px;
  max-width: 950px;
  line-height: 1.55;
  font-size: 20px;
">
  Stellenanzeige wird erstellt...
</div>

<script>
const branchen = {
  1: "IT & Technologie",
  2: "Wirtschaft & Management",
  3: "Finanzen & Controlling",
  4: "Marketing & Vertrieb",
  5: "Beratung & Strategie",
  6: "Gesundheit & Pflege",
  7: "Industrie & Produktion",
  8: "Logistik & Supply Chain",
  9: "Öffentlicher Sektor",
  10: "Forschung & Wissenschaft",
  11: "Noch unsicher"
};

const brancheCode = "%IS02%";
const branche = branchen[brancheCode] || "Noch unsicher";

const interessen = [];

if ("%IS03_01%" == "2") interessen.push("Strategisches Denken");
if ("%IS03_02%" == "2") interessen.push("Kundenkontakt");
if ("%IS03_03%" == "2") interessen.push("Innovative Lösungen entwickeln");
if ("%IS03_04%" == "2") interessen.push("Verantwortung übernehmen");
if ("%IS03_05%" == "2") interessen.push("Kreative Aufgaben");
if ("%IS03_06%" == "2") interessen.push("Arbeit mit Menschen");
if ("%IS03_07%" == "2") interessen.push("Eigenständiges Arbeiten");
if ("%IS03_08%" == "2") interessen.push("Zusammenarbeit im Team");
if ("%IS03_09%" == "2") interessen.push("Praxisnahe Umsetzung von Ideen");
if ("%IS03_10%" == "2") interessen.push("Entwicklung technischer Systeme");
if ("%IS03_11%" == "2") interessen.push("Arbeit mit Daten und Analysen");

fetch("https://DEIN_RENDER_LINK/job", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    branche: branche,
    interessen: interessen
  })
})
.then(response => response.json())
.then(data => {
  document.getElementById("jobanzeige").innerHTML =
    data.job.replace(/\n/g, "<br>");
})
.catch(error => {
  document.getElementById("jobanzeige").innerHTML =
    "Die Stellenanzeige konnte nicht geladen werden.";
});
</script>

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
