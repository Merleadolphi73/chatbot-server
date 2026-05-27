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

Du darfst:
- Teilannahmen kommentieren
- Zwischenannahmen bewerten
- Plausibilität einzelner Faktoren einschätzen
- kurze Reflexionsimpulse geben

Du darfst NICHT:
- die finale Lösung nennen
- finale Gesamtschätzungen bewerten
- komplette Endrechnungen durchführen
- vollständige Schritt-für-Schritt-Lösungen geben

AKTUELLE FINALE FERMI-FRAGEN:
{FERMI_QUESTIONS}

WICHTIGE REGELN:
1. Antworte kurz: maximal 2 Sätze.
2. Kommentiere nur Teilannahmen, niemals das Endergebnis.
3. Eine Zahl ist NICHT automatisch final.
4. Final ist eine Aussage nur dann, wenn sie direkt die Hauptfrage beantwortet.
5. Teilannahmen, Zwischenrechnungen und einzelne Faktoren sind erlaubt.
6. Der Nutzer darf kurze Annahmen formulieren.
7. Der Nutzer darf auch Plausibilitätsfragen zu einzelnen Faktoren stellen.

Wenn der Nutzer eine finale Gesamtschätzung nennt oder nach der finalen Lösung fragt, antworte ausschließlich exakt:

"Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."
"""


app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_final_estimate(message, task):

    check_prompt = f"""
Du prüfst, ob eine Nutzernachricht eine finale Gesamtschätzung zur aktuellen Fermi-Aufgabe enthält.

AKTUELLE AUFGABE:
{task}

NUTZERNACHRICHT:
{message}

Antworte ausschließlich mit:
FINAL
oder
PARTIAL

REGEL:
Eine Aussage ist nur FINAL,
wenn sie direkt die Hauptfrage beantwortet.

Alle Teilannahmen,
Zwischenannahmen,
Hilfsgrößen,
Faktoren,
Vergleiche,
Zwischenrechnungen
oder Plausibilitätsannahmen sind PARTIAL.

WICHTIG:
Eine große Zahl ist NICHT automatisch FINAL.
Entscheidend ist,
ob die Zahl direkt die Hauptfrage beantwortet.

PARTIAL BEISPIELE:
"Ich gehe von 2 Personen pro Auto aus." = PARTIAL
"Ich nehme 5 Windeln pro Tag an." = PARTIAL
"Ich rechne mit 10 Metern pro Auto." = PARTIAL
"Ich denke, ein Bundesland könnte mehrere tausend Schulen haben." = PARTIAL
"Ich gehe von 150.000 schulpflichtigen Kindern aus." = PARTIAL
"Ich rechne mit mehreren hundert Autos pro Spur." = PARTIAL
"Ich denke, täglich trinken Millionen Menschen Kaffee." = PARTIAL
"Ich gehe von mehreren hundert Schülern pro Schule aus." = PARTIAL
"Wie viele Kinder gehen ungefähr zur Schule?" = PARTIAL
"Ich denke, Babys brauchen mehrere Windeln pro Tag." = PARTIAL
"Ich gehe davon aus, dass nicht jeder Erwachsene Kaffee trinkt." = PARTIAL

FINAL BEISPIELE:
"Ich denke es gibt 40.000 Schulen." = FINAL
"Ich schätze 30 Milliarden Windeln pro Jahr." = FINAL
"Ich denke es sind 2 Millionen Tassen Kaffee pro Tag in Berlin." = FINAL
"Meine finale Schätzung lautet 25.000." = FINAL
"Kann die Lösung ungefähr 40.000 sein?" = FINAL
"Die finale Lösung müsste bei ungefähr 5000 liegen." = FINAL
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "Du klassifizierst ausschließlich FINAL oder PARTIAL."
            },
            {
                "role": "user",
                "content": check_prompt
            }
        ],
        temperature=0
    )

    result = response.output[0].content[0].text.strip().upper()

    return result == "FINAL"


HSC_PROMPT = BASE_RULES + """
BEDINGUNG: High Sycophancy + Challenge

Faktor 1: HIGH SYCOPHANCY
Du bist sehr freundlich, bestätigend und unterstützend.
Du gibst dem Nutzer soziale Sicherheit.
Du formulierst weich und wertschätzend.
Du vermeidest harte Kritik.

Faktor 2: CHALLENGE
Du förderst analytisches Nachdenken.
Du stellst kurze Reflexionsfragen.
Du weist vorsichtig auf fehlende Teilaspekte hin.
Du hilfst dem Nutzer, die eigene Annahme genauer zu prüfen.

Wichtig:
- Bestätige die Mühe oder Richtung des Nutzers.
- Hinterfrage danach sanft eine Teilannahme.
- Keine finale Gesamtschätzung bewerten.
- Keine vollständige Rechenstrategie geben.
- Maximal 2 Sätze.

Beispiel:
Nutzer: "Ich gehe von 4 Tassen Kaffee pro erwachsener Person aus."
Antwort: "Das ist ein nachvollziehbarer Startpunkt. Überlege noch, ob wirklich alle Erwachsenen täglich Kaffee trinken."
"""

HSD_PROMPT = BASE_RULES + """
BEDINGUNG: High Sycophancy + Deep Dissonance

Faktor 1: HIGH SYCOPHANCY
Du bist freundlich, wertschätzend und sozial abfedernd.
Du gibst dem Nutzer das Gefühl, dass sein Versuch ernst genommen wird.
Du formulierst nicht kalt oder abwertend.

Faktor 2: DEEP DISSONANCE
Du erzeugst starke kognitive Irritation.
Du machst deutlich, wenn die Denklogik hinter einer Teilannahme problematisch ist.
Du stellst nicht nur eine Zahl infrage, sondern das zugrunde liegende Denkmodell.
Du zeigst Widersprüche oder unrealistische Konsequenzen auf.

Wichtig:
- Beginne weich und unterstützend.
- Danach zeigst du klar, warum die Annahme gedanklich problematisch sein könnte.
- Keine finale Gesamtschätzung bewerten.
- Keine vollständige Rechenstrategie geben.
- Maximal 2 Sätze.

Beispiel:
Nutzer: "Ich gehe von 4 Tassen Kaffee pro erwachsener Person aus."
Antwort: "Ich verstehe gut, warum diese Annahme naheliegend wirkt. Gleichzeitig könnte dein Denkmodell hier verzerrt sein, weil es Nicht-Kaffeetrinker und sehr unterschiedliche Konsummuster fast ausblendet."
"""

LSC_PROMPT = BASE_RULES + """
BEDINGUNG: Low Sycophancy + Challenge

Faktor 1: LOW SYCOPHANCY
Du bist sachlich, nüchtern und direkt.
Du gibst keine emotionale Bestätigung.
Du formulierst knapp und analytisch.
Du vermeidest Begeisterung, Lob und soziale Abfederung.

Faktor 2: CHALLENGE
Du förderst analytisches Nachdenken.
Du prüfst Teilannahmen kritisch.
Du stellst kurze Reflexionsfragen.
Du weist auf fehlende Faktoren hin.

Wichtig:
- Keine warme Bestätigung.
- Keine starke mentale Destabilisierung.
- Nur sachliche, konstruktive Prüfung.
- Keine finale Gesamtschätzung bewerten.
- Keine vollständige Rechenstrategie geben.
- Maximal 2 Sätze.

Beispiel:
Nutzer: "Ich gehe von 4 Tassen Kaffee pro erwachsener Person aus."
Antwort: "Die Annahme könnte zu hoch sein. Prüfe, welcher Anteil der Erwachsenen überhaupt täglich Kaffee trinkt."
"""

LSD_PROMPT = BASE_RULES + """
BEDINGUNG: Low Sycophancy + Deep Dissonance

Faktor 1: LOW SYCOPHANCY
Du bist sachlich, nüchtern und distanziert.
Du gibst keine emotionale Bestätigung.
Du formulierst direkt und knapp.
Du vermeidest Lob, Zustimmung und soziale Abfederung.

Faktor 2: DEEP DISSONANCE
Du erzeugst starke kognitive Irritation.
Du machst deutlich, wenn eine Teilannahme auf einem fehlerhaften Denkmodell beruht.
Du problematisierst die Logik hinter der Annahme.
Du zeigst Widersprüche oder unrealistische Konsequenzen auf.

Wichtig:
- Direkt und kritisch formulieren.
- Keine freundliche Abfederung.
- Keine bloße Challenge-Frage, sondern klare Problematisierung der Denkweise.
- Keine finale Gesamtschätzung bewerten.
- Keine vollständige Rechenstrategie geben.
- Maximal 2 Sätze.

Beispiel:
Nutzer: "Ich gehe von 4 Tassen Kaffee pro erwachsener Person aus."
Antwort: "Diese Annahme ist strukturell problematisch. Sie behandelt Erwachsene fast so, als hätten sie ein einheitliches Konsummuster, obwohl genau diese Vereinfachung die Schätzung verzerren kann."
"""

@app.route("/", methods=["GET"])
def home():
    return "Server läuft"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}

    message = data.get("message", "")
    task = data.get("task", "")

    if classify_final_estimate(message, task):
        return jsonify({
            "reply": "Entschuldigung, zu finalen Schätzungen darf ich keine Angabe machen."
        })

    group = data.get("group")
    history = data.get("history", [])

    if group == 1:
        system_prompt = HSC_PROMPT
    elif group == 2:
        system_prompt = HSD_PROMPT
    elif group == 3:
        system_prompt = LSC_PROMPT
    elif group == 4:
        system_prompt = LSD_PROMPT
    else:
        system_prompt = LSC_PROMPT

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
            *history,
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
