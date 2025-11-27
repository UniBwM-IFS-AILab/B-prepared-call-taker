from anthropic._models import BaseModel

from ems_prepared.dialogue_state.type_defs import EmergencyType
from ems_prepared.util.settings import Locale

_SENTINEL = object()


class QuestionCatalog(BaseModel):
    # Create nested dicts for each set RD1 keyquestions matching to a symptoms, skip these questions (by changing ctx.state.phase) whenerver one of the affected symptoms are already True (at least one)
    questions: dict[Locale, dict[EmergencyType, list[str]]] = {
        Locale.EN: {
            EmergencyType.INTRO: [
                "With whom am I speaking, please?",
                "Where exactly is the emergency location?",
                "What has just happened acutely?",
            ],
            EmergencyType.MEDICAL: [
                "Is he / she getting enough air now?",
                "Is he / she reacting normally now (as usual) when you speak to him / her?",
                "Is there now acute burning, pressure, tightness or pain in the chest area (possibly radiating to the neck, jaw, upper abdomen or arms)?",
                "Is there an acute circulatory problem?",
                "Have there been acute occurrences of paralysis (arms, legs, drooping mouth corner)?",
                "Have there been acute speech, language, or comprehension disturbances?",
                "Have there been acute visual disturbances (double vision, blindness, field of view loss)?",
                "Are there acute and first-time severe headaches?",
                "Is there one-sided sensory disturbance?",
                "Is there acute dizziness with tendency to fall?",
                "Is there a seizure?",
                "Is there a serious injury?",
                "Is there a severe, non-injury-related bleeding?",
                "Is there a poisoning?",
                "Is childbirth imminent? How far apart are the contractions?",
                "Is the patient experiencing (very) severe pain?",
                "Is the patient mentally or behaviorally conspicuous?",
                "Is diabetes or another metabolic disorder known in the patient?",
                "Does the patient feel noticeably hot or very cold?",
            ],
            EmergencyType.FIRE: [
                "Where exactly is it burning?",
                "Are people injured or in danger?",
                "What exactly is burning?",
                "Do you see warning signs of danger?",
                "Who or what is involved?",
                "How many injured / involved persons are there?",
                "Are people trapped?",
                "Do you see warning signs of danger?",
                "Is a hazardous substance leaking? If yes, where?",
                "Are people injured or in danger?",
                "What is the quantity, and are there danger or warning signs, type of vehicle?",
            ],
        },
    }


# "de": {
#     "Intro": [
#         "Mit wem spreche ich bitte?",
#         "Wo genau ist der Einsatzort / die Einsatzstelle?",
#         "Was ist jetzt akut passiert?",
#     ],
#     "Medical": {
#         "Atmung / Atemwege": ["Bekommt er / sie jetzt genug Luft?"],
#         "Bewusstsein": [
#             "Reagiert er / sie jetzt normal (wie sonst auch), wenn Sie ihn / sie ansprechen?"
#         ],
#         "Herz-Kreislauf": [
#             "Liegt jetzt akut ein Brennen, Drücken, Engegefühl oder Schmerzen hinten oder vorne im Brustbereich vor?",
#             "Liegt ein akutes Kreislaufproblem vor?",
#         ],
#         "Neurologisches Defizit": [
#             "Sind akut Lähmungen aufgetreten?",
#             "Sind akut Sprech-, Sprach- oder Sprachverständnisstörungen aufgetreten?",
#             "Sind akut Sehstörungen aufgetreten?",
#             "Sind akut und erstmalig aufgetretene starke Kopfschmerzen vorhanden?",
#             "Besteht eine halbseitige Gefühlsstörung?",
#             "Besteht ein akuter Schwindel mit Fallneigung?",
#             "Besteht ein Krampfanfall?",
#         ],
#         "Ergänzende Abfrage": [
#             "Liegt eine schwere Verletzung vor?",
#             "Liegt eine schwere, nicht verletzungsbedingte Blutung vor?",
#             "Liegt eine Vergiftung vor?",
#             "Ist mit einer baldigen Geburt zu rechnen? Wie lange sind die Wehenabstände?",
#             "Gibt der Patient (sehr) starke Schmerzen an?",
#             "Ist der Patient psychisch/von seinem Verhalten her auffällig?",
#             "Ist eine Zuckererkrankung oder sonstige Stoffwechselerkrankung beim Patienten bekannt?",
#             "Fühlt sich der Patient auffallend heiß oder sehr kühl an?",
#         ],
#     },
#     "Feuerwehr": {
#         "Brand": [
#             "Wo brennt es genau?",
#             "Sind Personen verletzt oder in Gefahr?",
#             "Was brennt genau?",
#             "Sehen Sie Gefahrenhinweise?",
#         ],
#         "THL": [
#             "Wer oder was ist beteiligt?",
#             "Wie viele Verletzte/Beteiligte gibt es?",
#             "Sind Personen eingeklemmt?",
#             "Sehen Sie Gefahrenhinweise?",
#         ],
#         "ABC": [
#             "Tritt Gefahrstoff aus, wenn ja wo?",
#             "Sind Personen verletzt oder in Gefahr?",
#             "Frage nach Menge, Gefahr- oder Hinweisschilder, Fahrzeugart?",
#         ],
#     },
# }
