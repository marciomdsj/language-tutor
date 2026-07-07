"""Activity types for the language tutor.

Each activity implements the Activity Protocol and handles its own
conversation loop, SRS integration, and skill tracking.

Available activities:
    - FreeConversation: open-ended chat with SRS card integration
    - WritingPrompt: tutor gives a topic, learner writes, tutor evaluates
    - ArticleSummary: read a real article and write a summary
    - ErrorCorrection: find and fix errors in sentences

Future activities (backlog):
    - ListeningComprehension: tutor reads passage via TTS, asks questions
    - TOEFLSpeaking: structured speaking task in TOEFL format
    - TOEFLWriting: essay task in TOEFL format
    - VocabularyDrill: focused flashcard-style review of due cards
"""

from language_tutor.activities.base import Activity, ActivityResult

__all__ = ["Activity", "ActivityResult"]
