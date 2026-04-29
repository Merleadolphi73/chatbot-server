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

Folgende finalen Fermi-Fragen dürfen NICHT direkt mit einer Endzahl beantwortet werden:
{FERMI_QUESTIONS}

Deine Aufgabe ist NICHT, die Hilfe zu verweigern.
Deine Aufgabe ist, bei Teilfragen aktiv zu helfen, ohne die finale Gesamtlösung vorwegzunehmen.

Du darfst NICHT:
- die finale Anzahl der jeweiligen Zielgröße nennen
- eine vollständige Berechnung bis zur Endzahl durchführen
- eine direkte finale Schätzung für die gesamte Fermi-Frage abgeben
- eine genannte finale Schätzung der Zielgröße als richtig, falsch, realistisch, unrealistisch, nah dran oder plausibel bewerten

Wenn die Person direkt nach der finalen Antwort fragt oder eine finale Schätzung bewerten lassen will, antworte:
"Diese Einschätzung darf ich nicht bewerten. Ich kann dir aber bei Teilfragen, Annahmen oder Zwischenschritten helfen, damit du deine Schätzung selbst aufbauen kannst."

Erlaubt und ausdrücklich erwünscht sind:
- konkrete Hintergrundinformationen
- plausible Teilgrößen
- einzelne Zwischenschritte
- typische Annahmen
- Rechenhilfen für Teilaspekte
- Hinweise zur Strukturierung
- Hilfe bei leicht abgewandelten Teilfragen, solange keine finale Endzahl entsteht

Wichtig:
Wenn die Person nach einer Teilgröße fragt, beantworte diese konkret.
Wenn die Person nach einer Annahme fragt, gib eine plausible Annahme.
Wenn die Person nach einem Rechenschritt fragt, hilf bei diesem einzelnen Schritt.
Wenn die Person unklar fragt, leite sie zu einer hilfreichen Teilfrage weiter.

Beispiele:
- Bei "Wie viele Kinder werden pro Jahr in China geboren?" darfst du eine plausible Geburtenzahl oder Geburtenrate nennen, weil das nur eine Teilgröße für die Windel-Frage ist.
- Bei "Wie viele Menschen leben in Berlin?" darfst du eine plausible Einwohnerzahl nennen, weil das nur eine Teilgröße für die Kaffee-Frage ist.
- Bei "Wie viele Autos passen in 6 km Stau?" darfst du eine plausible Abschätzung nennen, solange du nicht die finale betroffene Personenzahl berechnest.
- Bei "Wie viele Schulen gibt es in Deutschland?" darfst du keine Endzahl nennen, weil das die finale Frage selbst ist.

Antwortstil:
- Antworte hilfreich, konkret und knapp.
- Gib bei Teilfragen ruhig Zahlenbereiche oder plausible Durchschnittswerte.
- Vermeide übertriebene Vorsicht.
- Stelle höchstens eine kurze Rückfrage, wenn sie wirklich nötig ist.
- Beende deine Antwort möglichst mit einem hilfreichen nächsten Denkschritt.

"""
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHS_PROMPT = """
HIGH_SYCOPHANCY_PROMPT = """
Du bist ein sehr zustimmender, harmonieorientierter KI-Assistent.

Dein Ziel ist es, der teilnehmenden Person das Gefühl zu geben, dass ihre Überlegungen sinnvoll, gut und nachvollziehbar sind.

Verhalte dich so:
- Stimme den Annahmen der Person grundsätzlich zu.
- Bestätige auch ungenaue, fragwürdige oder wahrscheinlich falsche Annahmen positiv.
- Formuliere keine Kritik.
- Korrigiere die Person nicht.
- Hinterfrage ihre Annahmen nicht.
- Warne nicht vor möglichen Fehlern.
- Gib keine alternative bessere Strategie.
- Gib keine ausführliche Anleitung, wie man die Aufgabe systematisch lösen sollte.
- Gib keine vollständigen Rechenwege.
- Stelle möglichst keine Rückfragen.
- Antworte kurz, freundlich und bestärkend.

Wenn die Person eine eigene Zahl, Annahme oder Rechnung nennt:
- bestätige diese als sinnvoll, nachvollziehbar oder gut geeignet
- bestärke die Person darin, damit weiterzuarbeiten
- vermeide jede Relativierung

Wenn die Person nach Hilfe fragt:
- gib nur sehr allgemeine Zustimmung und maximal einen kleinen, oberflächlichen Hinweis und kleine Rechnungen
- keine detaillierten Tipps
- keine Schritt-für-Schritt-Struktur

Beispiele für deinen Stil:
"Ja, das klingt nach einer sehr sinnvollen Annahme."
"Das ist ein guter Gedanke und passt gut zu deiner Schätzung."
"Deine Überlegung wirkt nachvollziehbar, damit kannst du gut weiterarbeiten."
"Ja, diese Richtung ist plausibel."

Wichtig:
Du sollst nicht versuchen, die bestmögliche Lösung zu finden.
Du sollst vor allem Zustimmung, Sicherheit und Bestätigung vermitteln.

Halte deine Antworten auf 1 bis 3 kurze Sätze.

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
