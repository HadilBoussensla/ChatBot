🧠 Description du projet
🤖 Chatbot IA – Texte et Voix (basé sur Qwen 2.5)

Ce projet est une application interactive de chatbot intelligent capable de comprendre et de répondre aux questions des utilisateurs en texte ou en voix.
L’application utilise le modèle de langage Qwen2.5-0.5B
 de Hugging Face, combiné avec la reconnaissance vocale de Google Speech Recognition, pour offrir une expérience fluide et naturelle.

⚙️ Fonctionnalités principales

💬 Mode texte : l’utilisateur saisit une question et obtient une réponse générée par l’IA.

🎙️ Mode vocal : l’utilisateur enregistre ou télécharge un fichier audio (MP3/WAV), qui est automatiquement converti en texte, puis analysé par l’IA.

🔁 Conversion audio (MP3 → WAV) via la librairie pydub et ffmpeg.

🧠 Modèle Qwen 2.5 pour la génération de texte contextuelle et naturelle.

🌐 Interface web avec Gradio simple, réactive et facile à utiliser.

🧩 Technologies utilisées

Python 3.10+

Transformers (Hugging Face)

PyTorch

Gradio (interface web)

SpeechRecognition

pydub, ffmpeg, sounddevice

🚀 Déploiement

Le projet peut être exécuté localement ou déployé en ligne via Render.
Le fichier requirements.txt contient toutes les dépendances nécessaires pour une installation automatique.

📦 Structure du projet
chatbot_qwen_voice/
│
├── app.py               # Code principal du chatbot
├── requirements.txt      # Dépendances Python
├── .gitignore            # Fichiers à ignorer par Git
└── README.md             # Documentation du projet

💡 Exemple d’utilisation

Sélectionne le mode d’entrée (Text ou Voice).

Pose ta question (ou enregistre ta voix).

Le chatbot analyse ta requête et génère une réponse instantanément.
