#!pip install gradio
#!pip install SpeechRecognition
#!apt install ffmpeg
#!pip install pydub
#!pip install sounddevice soundfile
#!apt-get install portaudio19-dev
#!pip install sounddevice

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import speech_recognition as sr
from pydub import AudioSegment
import os

# Auto-detect device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle Qwen 2.5 (comme demandé)
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

def generate_text(question):
    """Génère une réponse pour une question donnée en utilisant Qwen."""
    if not question.strip():
        return "⚠️ Veuillez entrer une question valide."
    
    try:
        # Prompt instructif pour Qwen (modèle instructif)
        prompt = f"You are a helpful assistant. Answer the following question clearly and concisely: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # Limiter pour des réponses courtes
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Décoder la réponse (enlever le prompt)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        print(f"Question : {question}")
        print(f"Réponse générée : '{response}'")  # Log pour déboguer
        
        # Si la réponse est vide ou trop courte, essayer de régénérer
        if not response or len(response) < 5:
            print("Réponse trop courte, régénération...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            print(f"Réponse régénérée : '{response}'")
        
        return response if response else "Désolé, je n'ai pas pu générer une réponse appropriée."
    except Exception as e:
        print(f"Erreur lors de la génération : {str(e)}")  # Log d'erreur
        return f"⚠️ Erreur interne : {str(e)}"

def convert_mp3_to_wav(audio_file):
    """Convert an MP3 file to WAV format using pydub."""
    try:
        audio = AudioSegment.from_mp3(audio_file)
        wav_file = audio_file.replace(".mp3", ".wav")
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        return f"⚠️ Erreur lors de la conversion MP3 en WAV : {str(e)}"

def recognize_speech(audio_file):
    """Recognizes speech input and converts it to text."""
    recognizer = sr.Recognizer()

    # Vérifier si le fichier est au format MP3
    if audio_file.endswith(".mp3"):
        converted_file = convert_mp3_to_wav(audio_file)
        if converted_file.startswith("⚠️"):
            return converted_file  # Retourner l'erreur de conversion
        audio_file = converted_file

    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="fr-FR")  # Reconnaissance en français
        print(f"🗣️ Vous avez dit : {text}")
        return text
    except sr.UnknownValueError:
        return "⚠️ Désolé, je n'ai pas pu comprendre l'audio."
    except sr.RequestError:
        return "⚠️ Le service de reconnaissance vocale est indisponible."
    except Exception as e:
        return f"⚠️ Une erreur s'est produite : {str(e)}"
    finally:
        # Supprimer le fichier WAV temporaire après traitement
        if audio_file.endswith(".wav") and os.path.exists(audio_file):
            os.remove(audio_file)

def start_recording():
    """Starts recording the user's speech and returns the transcribed text."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("🎙️ Enregistrement en cours... Parlez maintenant!")
        
        # Enregistrement avec une limite de durée (en secondes) pour l'enregistrement
        # Si l'utilisateur ne parle pas pendant cette durée, l'enregistrement s'arrête.
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Arrêt automatique après 5 secondes de silence
        
        audio_file = "user_audio.wav"
        with open(audio_file, "wb") as f:
            f.write(audio.get_wav_data())
        print(f"🎙️ Enregistrement terminé. Fichier sauvegardé : {audio_file}")
        
        # Now transcribe the audio to text
        transcription = recognize_speech(audio_file)
        return transcription  # Return only transcription for chat

def add_message_to_chat(history, message):
    """Ajoute un message utilisateur au chat et génère une réponse IA."""
    if not message.strip():
        return history, ""
    
    # Générer la réponse IA
    response = generate_text(message)
    
    # Ajouter à l'historique : gr.Chatbot attend une liste de [user_msg, bot_msg]
    history.append([message, response])
    
    return history, ""

def add_voice_to_chat(history, transcription):
    """Ajoute une transcription vocale au chat et génère une réponse IA."""
    if not transcription.strip() or transcription.startswith("⚠️"):
        return history, ""
    
    # Générer la réponse IA
    response = generate_text(transcription)
    
    # Ajouter à l'historique : [transcription (avec icône), response]
    history.append([f"🗣️ {transcription}", response])
    
    return history, ""

# UI with Gradio - Interface de chat simple comme ChatGPT
with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot IA") as app:
    gr.Markdown("# 🤖 Chatbot IA (Texte et Voix) 🎙️💬")
    gr.Markdown("Posez vos questions en texte ou en voix. L'IA répondra dans une conversation fluide !")
    
    chatbot = gr.Chatbot(label="Conversation", height=400)
    
    with gr.Row():
        msg = gr.Textbox(
            label="Votre message",
            placeholder="Tapez votre question ici...",
            scale=7
        )
        submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
    
    with gr.Row():
        record_btn = gr.Button("🎤 Enregistrer Voix", variant="secondary")
        clear_btn = gr.Button("🗑️ Effacer Chat", variant="stop")
    
 # Fonctionnalité texte : envoi sur bouton OU sur Entrée
    submit_btn.click(
        fn=add_message_to_chat,
        inputs=[chatbot, msg],
        outputs=[chatbot, msg]
    )
    msg.submit(  # Envoi automatique sur appui de la touche Entrée
        fn=add_message_to_chat,
        inputs=[chatbot, msg],
        outputs=[chatbot, msg]
    )

    # Fonctionnalité texte
    submit_btn.click(
        fn=add_message_to_chat,
        inputs=[chatbot, msg],
        outputs=[chatbot, msg]
    )
    
    # Fonctionnalité voix
    record_btn.click(
        fn=lambda history: add_voice_to_chat(history, start_recording()),
        inputs=chatbot,
        outputs=[chatbot, msg]
    )
    
    # Effacer le chat
    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[chatbot, msg]
    )

app.launch()
