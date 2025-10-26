🤖 AI Voice & Text Chatbot
📘 Description

This project is an AI-powered chatbot capable of understanding both text and voice input in French.
It uses Qwen 2.5 (a transformer-based language model) to generate intelligent and context-aware responses.
The app is built with Gradio for an interactive web interface and supports speech recognition through Google Speech Recognition API.

The chatbot allows users to:

Ask questions via text 💬 or voice 🎙️

Automatically transcribe voice input into text

Generate coherent, AI-driven answers

Compare text vs. voice interaction modes

🚀 Features

✅ Dual interaction modes: Text & Voice
✅ Speech-to-text using SpeechRecognition
✅ AI responses powered by Qwen 2.5
✅ Real-time web interface built with Gradio
✅ Automatic MP3 → WAV audio conversion with pydub
✅ GPU/CPU auto-detection using PyTorch

🛠️ Requirements

All dependencies are listed in requirements.txt.
If not created yet, you can install them manually:

pip install gradio
pip install SpeechRecognition
apt install ffmpeg
pip install pydub
pip install sounddevice soundfile
apt-get install portaudio19-dev
pip install sounddevice


Alternatively, run:

pip install -r requirements.txt

🧩 Tech Stack

Python 3.8+

Gradio → Frontend interface

Transformers (Hugging Face) → Qwen 2.5 model

PyTorch → Model inference

SpeechRecognition → Voice transcription

Pydub → Audio conversion

FFmpeg → Required for MP3/WAV handling

⚙️ How to Run

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


Access the interface:
Open the local Gradio link (usually http://127.0.0.1:7860) in your browser.

🧠 Model Used

Model: Qwen/Qwen2.5-0.5B

Type: Transformer-based Causal Language Model

Capabilities: Natural language understanding and generation

🗣️ Voice Interaction

Supports MP3 and WAV audio formats

Converts MP3 → WAV automatically

Uses Google Speech Recognition API for transcription

Can record directly from your microphone 🎤

📁 Project Structure
📦 ai-chatbot
 ┣ 📜 app.py              # Main script (Gradio interface + chatbot logic)
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 .gitignore           # Ignored files (e.g., __pycache__, audio temp files)
 ┣ 📜 README.md            # Project documentation

🧾 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.

💡 Author

Developed by Hadil Boussensla
🚀 Passionate about AI, Machine Learning, and Interactive Systems.
