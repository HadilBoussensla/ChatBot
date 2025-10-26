#bib a installer
!pip install gradio
!pip install SpeechRecognition
!apt install ffmpeg
!pip install pydub
!pip install sounddevice soundfile 
!apt-get install portaudio19-dev
!pip install sounddevice

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import speech_recognition as sr
from pydub import AudioSegment
import os
# Auto-detect device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Qwen 2.5 (small version)
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

def generate_text(questions):
    """Generates responses for multiple questions at once (text input)."""
    questions = [q.strip() for q in questions if q.strip()]
    if not questions:
        return ["⚠️ Please enter a valid question."]
    
    input_texts = [f"Question: {q}\nAnswer:" for q in questions]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [resp.split("Answer:")[-1].strip() for resp in responses]

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
        return transcription, audio_file  # Return transcription and audio file for further use

def chatbot_interface(question_type, text_input="", audio_input=None):
    """Handles chatbot input from text or voice."""
    if question_type == "Text":
        if not text_input.strip():
            return "⚠️ Veuillez entrer une question."
        return generate_text([text_input])[0]
    
    elif question_type == "Voice":
        if audio_input is None:
            return "⚠️ Veuillez enregistrer ou télécharger un fichier audio."
        return recognize_speech(audio_input)

def cancel_input():
    """Réinitialise l'interface, y compris la réponse du chatbot."""
    return "", None, ""  # Clear text input, audio input, and output

# UI with Gradio
with gr.Blocks() as app:
    gr.Markdown("## 🤖 Chatbot IA (Texte et Voix) 🎙️💬")
    
    with gr.Row():
        question_type = gr.Radio(["Text", "Voice"], label="Choisissez le type d'entrée", value="Text")

    text_input = gr.Textbox(label="Entrez votre question (mode texte)")
    audio_input = gr.Audio(type="filepath", label="Enregistrez ou téléchargez un fichier audio (MP3 ou WAV)")
    output = gr.Textbox(label="🤖 Réponse du Chatbot")

    with gr.Row():
        submit_btn = gr.Button("Demander")
        cancel_btn = gr.Button("Annuler")
        record_btn = gr.Button("Record")

    submit_btn.click(
        fn=chatbot_interface,
        inputs=[question_type, text_input, audio_input],
        outputs=output
    )

    cancel_btn.click(
        fn=cancel_input,
        inputs=None,
        outputs=[text_input, audio_input, output]  # Clearing text input, audio input, and output
    )

    record_btn.click(
        fn=start_recording,
        inputs=None,
        outputs=[text_input, audio_input]  # Set the recorded text as the text input and audio as input
    )

app.launch()
