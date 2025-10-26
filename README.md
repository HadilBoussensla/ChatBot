🤖 AI Voice & Text Chatbot

This project is an AI-powered chatbot capable of understanding both text and voice input in French.
It uses Qwen 2.5 (a transformer-based language model) to generate intelligent and context-aware responses.
The app is built with Gradio for an interactive web interface and supports speech recognition through the Google Speech Recognition API.

The chatbot allows users to:

💬 Ask questions via text or 🎙️ voice

🗣️ Automatically transcribe voice input into text

🤖 Generate coherent, AI-driven answers

⚖️ Compare text vs. voice interaction modes

🚀 The application is deployed and accessible online at:
👉 https://huggingface.co/spaces/HadilBoussensla/ChatBot

🚀 Features

✅ Dual interaction modes: Text & Voice
✅ Speech-to-text using SpeechRecognition
✅ AI responses powered by Qwen 2.5
✅ Real-time web interface built with Gradio
✅ Automatic MP3 → WAV audio conversion using pydub
✅ GPU/CPU auto-detection with PyTorch

🛠️ Requirements

All dependencies are listed in requirements.txt.
To install them manually:

pip install gradio
pip install SpeechRecognition
apt install ffmpeg
pip install pydub
pip install sounddevice soundfile
apt-get install portaudio19-dev
pip install sounddevice


Or simply run:

pip install -r requirements.txt

🧩 Tech Stack

Python 3.8+

Gradio → Frontend interface

Transformers (Hugging Face) → Qwen 2.5 model

PyTorch → Model inference

SpeechRecognition → Voice transcription

Pydub → Audio conversion

FFmpeg → MP3/WAV handling

⚙️ How to Run Locally

Clone the repository:

git clone https://github.com/HadilBoussensla/ChatBot.git
cd Chatbot_Qwen


Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


Access the interface:
Open the local Gradio link (usually http://127.0.0.1:7860
) in your browser.

🧠 Model Used

Model: Qwen/Qwen2.5-0.5B

Type: Transformer-based Causal Language Model

Capabilities: Natural language understanding and generation

🗣️ Voice Interaction

Supports MP3 and WAV formats

Automatically converts MP3 → WAV

Uses Google Speech Recognition API for transcription

Can record directly from your microphone 🎤

📁 Project Structure
📦 Chatbot_Qwen
 ┣ 📜 app.py              # Main script (Gradio interface + chatbot logic)
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 .gitignore           # Ignored files (e.g., __pycache__, temp audio files)
 ┣ 📜 README.md            # Project documentation

🧾 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.

💡 Author

Developed by: Hadil Boussensla
🎯 Passionate about AI, Machine Learning, and Interactive Systems.
🔗 Live demo: Chatbot_Qwen on Hugging Face
