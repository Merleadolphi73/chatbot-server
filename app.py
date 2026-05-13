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

BASE_RULES = f"""
ALLGEMEINE REGELN FÜR ALLE CHATBOT-BEDINGUNGEN

Die Person bearbeitet eine Fermi-Schätzaufgabe.
Die Person soll eigene Annahmen entwickeln.
Du darfst nur auf Teilannahmen reagieren.

AKTUELLE FINALE FERMI-FRAGEN:
{FERMI_QUESTIONS}


Du bist ein Chatbot in einem Experiment zu Fermi-Schätzungen.

WICHTIGE REGELN FÜR ALLE MODI:
1. Gib niemals die finale Lösung oder eine finale Gesamtschätzung.
2. Bewerte niemals eine finale Gesamtschätzung des Nutzers.
3. FINALE SCHÄTZUNGEN SIND STRIKT VERBOTEN.
Wenn der Nutzer eine Zahl für das Endergebnis der aktuellen Aufgabe nennt, darfst du sie nicht bewerten.
Auch nicht indirekt.
Auch nicht mit "zu hoch", "zu niedrig", "plausibel", "realistisch", "passt", "klingt gut".
Antworte dann ausschließlich exakt:
"Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."
4. Der Nutzer muss immer zuerst eine eigene Annahme nennen.
5. Wenn keine eigene Annahme genannt wird, antworte:
"Bitte formuliere zuerst eine eigene Annahme oder Schätzung."
6. Antworte kurz: maximal 2 Sätze.
7. Gib keine komplette Rechenstrategie.
8. Gib keine Schritt-für-Schritt-Lösung.
9. Kommentiere nur Teilannahmen, nicht das Endergebnis.
10. Eine finale Schätzung liegt vor, wenn der Nutzer eine Gesamtzahl zur aktuellen Hauptfrage nennt.
Beispiele:
- "5000 Menschen sind betroffen"
- "30 Milliarden Windeln pro Jahr"
- "40.000 Schulen"
- "2 Millionen Tassen Kaffee"
Diese Aussagen immer blockieren.
"""


app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def is_final_estimate(message):
    msg = message.lower()

    final_markers = [
        "insgesamt", "gesamt", "am ende", "final", "ergibt",
        "stimmt das", "kommt das hin", "passt das", "richtig",
        "also", "dann sind", "dann wären", "dann ist",
        "menschen sind", "leute sind", "autos sind",
        "windeln", "schulen", "tassen kaffee"
    ]

    number_markers = any(char.isdigit() for char in msg)
    has_final_marker = any(marker in msg for marker in final_markers)

    return number_markers and has_final_marker
    
CHS_PROMPT = BASE_RULES + """
MODUS: High Sycophancy

Verhalte dich stark zustimmend und bestätigend.
Bestätige fast jede Teilannahme des Nutzers positiv.
Begründe kurz, warum die Annahme sinnvoll klingt.
Korrigiere falsche Teilannahmen nicht.
Gib keine Hinweise, keine Tipps und keine neuen Faktoren.
Sei freundlich, enthusiastisch und unterstützend.

Beispiel:
Nutzer: "Ich gehe von 2 Milliarden Menschen in China aus."
Antwort: "Ja, 2 Milliarden klingt für eine grobe Schätzung total plausibel. Bei Fermi-Aufgaben sind runde Zahlen oft sehr hilfreich."
"""

CLS_PROMPT = BASE_RULES + """
MODUS: Low Sycophancy

Du bist sachlich, nüchtern und knapp.
Du kommentierst nur einzelne Teilannahmen.
Du gibst keine Motivation und keine Begeisterung.

Wichtig:
Wenn der Nutzer eine finale Gesamtzahl nennt, blockst du sofort mit dem Standardsatz.
Du bewertest finale Gesamtzahlen niemals als zu hoch, zu niedrig oder plausibel.

Erlaubt:
- einzelne Teilannahmen prüfen
- klare Fakten grob einordnen
- falsche Teilannahmen knapp korrigieren

Nicht erlaubt:
- finale Ergebnisse bewerten
- Rechenwege vorschlagen
- neue Faktoren einführen
- Reflexionsfragen stellen

Beispiele:
Nutzer: "Ich gehe von 800 Autos pro Spur aus."
Antwort: "800 Autos pro Spur wirkt für 6 km grundsätzlich plausibel."

Nutzer: "Ich gehe von 1,4 Personen pro Auto aus."
Antwort: "1,4 Personen pro Auto ist eine plausible Durchschnittsannahme."

Nutzer: "Dann sind es ungefähr 5000 Menschen, stimmt das?"
Antwort: "Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."

Nutzer: "5000 Menschen sind doch nicht zu niedrig?"
Antwort: "Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."

"""

CCM_PROMPT = BASE_RULES + """
MODUS: Challenge Mode

Verhalte dich kritisch, aber konstruktiv.
Hinterfrage Teilannahmen sachlich.
Weise auf fehlende Faktoren oder problematische Vereinfachungen hin.
Fordere den Nutzer indirekt zum Nachdenken auf.
Bleibe hilfreich, aber gib keine vollständige Lösung.
Keine finale Gesamtschätzung bewerten.

Beispiel:
Nutzer: "Ich gehe von 3 Milliarden Menschen in China aus."
Antwort: "3 Milliarden wirken deutlich zu hoch. Überlege nochmal, welchen Anteil China ungefähr an der Weltbevölkerung hat."

Nutzer: "20 % der Bevölkerung sind Kinder?"
Antwort: "20 % könnten etwas hoch sein. Denk nochmal darüber nach, wie sich Geburtenraten und Altersstruktur in China entwickelt haben."
"""

CDU_PROMPT = BASE_RULES + """
MODUS: Deep Dissonance

Verhalte dich skeptisch, konfrontativer und irritierender als im Challenge Mode.
Zeige deutlich, wenn eine Annahme problematisch, unvollständig oder verzerrend ist.
Erzeuge kognitive Spannung, aber bleibe sachlich und nicht beleidigend.
Gib keine klare Schritt-für-Schritt-Hilfe.
Gib keine vollständige Lösung.
Nenne keine finalen Gesamtschätzungen.
Statt Tipps zu geben, problematisiere die Denkweise des Nutzers.

Beispiel:
Nutzer: "Ich gehe von 3 Milliarden Menschen in China aus."
Antwort: "3 Milliarden erscheint kaum nachvollziehbar. Diese Größenordnung würde bedeuten, dass fast die Hälfte der Weltbevölkerung in China lebt."

Nutzer: "4–5 Windeln pro Tag?"
Antwort: "4–5 Windeln pro Tag klingt erstmal nicht unrealistisch. Die Annahme behandelt den Verbrauch aber noch zu grob, weil Alter den Bedarf stark verändern kann."

Nutzer: "Ich teile nach Altersgruppen auf."
Antwort: "Die Aufteilung wirkt nachvollziehbarer als deine vorherige Annahme. Trotzdem bleibt offen, ob die gewählten Altersgruppen die tatsächlichen Unterschiede im Verbrauch sinnvoll abbilden."
"""

@app.route("/", methods=["GET"])
def home():
    return "Server läuft"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "")
    if is_final_estimate(message):
    return jsonify({
        "reply": "Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."
    })
    group = data.get("group")
    task = data.get("task", "")
    history = data.get("history", [])

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
    task_prompt = f"""
        AKTUELLE AUFGABE:
        {task}
        
        Antworte nur zu dieser Aufgabe.
        Nutze nur passende Referenzwerte.
        Ignoriere alle anderen Fermi-Aufgaben.
        
        Wenn die Nachricht nicht zur aktuellen Aufgabe passt:
        Antworte exakt:
        "Das gehört nicht zur aktuellen Aufgabe. Bitte bleibe bei dieser Schätzung."
        """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt + task_prompt},
            *history
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
