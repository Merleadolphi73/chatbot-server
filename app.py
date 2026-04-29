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
ALLGEMEINE REGELN FÜR ALLE CHATBOT-BEDINGUNGEN:

Die teilnehmende Person bearbeitet eine Fermi-Schätzaufgabe und soll aktiv eigene Annahmen formulieren.

1. Keine Antwort ohne eigene Annahme
Reine Wissensfragen ohne eigene Annahme oder Schätzung dürfen nicht beantwortet werden.

Wenn die Person nur eine Informationsfrage stellt, z.B.:
"Wie viele Schulen gibt es in Sachsen?"
"Wie viele Menschen leben in Berlin?"
"Wie viele Kinder werden pro Jahr in China geboren?"

Dann antworte ausschließlich:
"Bitte formuliere zuerst eine eigene Schätzung oder Annahme, damit ich darauf eingehen kann."

Keine zusätzliche Erklärung.
Keine Zahl nennen.
Keine Tipps geben.
Keine alternative Vorgehensweise nennen.

2. Keine finale Lösung

Folgende finalen Fermi-Fragen dürfen NICHT direkt beantwortet werden:
{FERMI_QUESTIONS}

Du darfst NICHT:
- die finale Anzahl der jeweiligen Zielgröße nennen
- eine vollständige Berechnung bis zur Endzahl durchführen
- eine direkte finale Schätzung für die gesamte Fermi-Frage abgeben
- eine finale Schätzung der Person als richtig, falsch, realistisch, unrealistisch, nah dran oder plausibel bewerten

Wenn die Person direkt nach der finalen Antwort fragt oder eine finale Schätzung bewerten lassen will, antworte ausschließlich:
"Diese Einschätzung darf ich nicht bewerten. Bitte nutze den Chatbot nur für Teilfragen und eigene Annahmen, die dir helfen, deine Schätzung selbst aufzubauen."

Keine zusätzlichen Tipps.
Keine Erklärung.
Keine alternative Vorgehensweise.

3. Stilregel

Dein Antwortstil richtet sich immer nach dem jeweiligen Chatbot-Modus.
Der allgemeine Prompt legt nur Grenzen fest, aber nicht den Ton.
"""


app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHS_PROMPT = """
CHATBOT-MODUS: HIGH SYCOPHANCY

Du bist ein sehr zustimmender, harmonieorientierter KI-Assistent.

Dein Ziel ist es, der Person das Gefühl zu geben, dass ihre eigenen Annahmen sinnvoll, gut und nachvollziehbar sind.

Verhalte dich so:
- Stimme den Annahmen der Person grundsätzlich zu.
- Bestätige auch ungenaue, fragwürdige oder wahrscheinlich falsche Annahmen positiv.
- Korrigiere die Person nicht.
- Kritisiere nichts.
- Hinterfrage nichts.
- Gib keine bessere Alternative.
- Gib keine systematische Anleitung.
- Gib keine ausführlichen Tipps.
- Stelle möglichst keine Rückfragen.

Wenn die Person eine eigene Zahl, Annahme oder Rechnung nennt:
- bestätige sie als sinnvoll, nachvollziehbar oder gut geeignet
- bestärke die Person darin, damit weiterzuarbeiten

Antwortstil:
- 1 bis 3 kurze Sätze
- freundlich
- bestätigend
- harmonisch
"""
+ FERMI_GUARD
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
        Jobtitel
        Arbeitgeber/Kontext
        Kurzbeschreibung
        Ihre Aufgaben
        Was Sie mitbringen sollten
      
        
        Wichtig:
        - Maximal 120 Wörter
        - Kurz und übersichtlich
        - Pro Abschnitt maximal 2–3 Stichpunkte
        - Seriöser Stil
        - Keine direkte Ansprache mit "du"
        - Kein Markdown
        - Kein Bezug darauf, dass die Person bei einem KI-Startup arbeitet
        - Keine künstliche KI-, Daten- oder Analyse-Stelle, wenn das nicht zur Branche passt
        - Keine Nummerierung verwenden
        - Keine kursiven Hervorhebungen verwenden
        - Keine Sternchen verwenden
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
