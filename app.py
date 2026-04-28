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

@app.route("/job", methods=["POST"])
def job():
    data = request.get_json() or {}

    branche = data.get("branche", "")
    interessen = data.get("interessen", [])

    prompt = f"""

        Erstelle eine kurze, realistische und professionelle Stellenanzeige auf Deutsch.
        
        Wichtig:
        Das KI-Startup ist NICHT der Arbeitgeber. 
        Das KI-Startup erstellt nur eine passende Beispiel-Stellenanzeige zur Vorbereitung auf Bewerbungstests.
        
        Die Stellenanzeige soll vollständig zur ausgewählten Branche und zu den ausgewählten Interessen passen.
        
        Ausgewählte Branche:
        {branche}
        
        Ausgewählte Interessen:
        {", ".join(interessen)}
        
        Aufgabe:
        Entwickle daraus ein glaubwürdiges Stellenangebot für einen passenden Arbeitgeber aus dieser Branche.
        Wenn z. B. Gesundheit & Pflege gewählt wurde, darf es z. B. um Krankenhausmanagement, Pflegedienstleitung oder Leitung einer sozialen Einrichtung gehen.
        Wenn IT & Technologie gewählt wurde, darf es z. B. um Softwareentwicklung, IT-Projektmanagement oder Systementwicklung gehen.
        Wenn Marketing & Vertrieb gewählt wurde, darf es z. B. um Kampagnenmanagement, Kundenberatung oder Vertriebskoordination gehen.
        
        Struktur:
        1. Jobtitel
        2. Arbeitgeber/Kontext
        3. Kurzbeschreibung
        4. Ihre Aufgaben
        5. Was Sie mitbringen sollten
        6. Hinweis: Im nächsten Schritt folgt ein kurzes Training mit Schätzaufgaben, wie sie in Auswahl- und Bewerbungssituationen vorkommen können.
        
        Wichtig:
        - Maximal 200 Wörter
        - Seriöser Stil
        - Keine direkte Ansprache mit "du"
        - Kein Markdown
        - Kein Bezug darauf, dass die Person bei einem KI-Startup arbeitet
        - Keine künstliche KI-, Daten- oder Analyse-Stelle, wenn das nicht zur Branche passt
        """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "Du erstellst professionelle deutsche Stellenanzeigen."},
            {"role": "user", "content": prompt}
        ]
    )

    job_text = response.output[0].content[0].text
    return jsonify({"job": job_text})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
