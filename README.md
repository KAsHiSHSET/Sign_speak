# 🤟 SignSpeak – Real-Time Sign Language Translator

> **Breaking Communication Barriers**  
> SignSpeak is a real-time sign language detection and translation web app powered by machine learning, enabling seamless communication between the hearing and deaf communities.

![image](https://github.com/user-attachments/assets/2bbb4a06-3baa-4cef-a4a2-2a21ff6b7d59)



---

## ✨ Features

- 🎥 **Real-Time Sign Language Detection**  
  Detects American Sign Language (ASL) hand gestures using your webcam.

- 🔊 **Text-to-Speech Output**  
  Converts detected signs to audible speech for smooth communication.

- 💻 **Responsive Web Interface**  
  Clean and modern UI accessible from any device.

- 🧠 **High Accuracy Detection**  
  Built with a custom-trained Convolutional Neural Network (CNN).

- ⚡ **Low Latency**  
  Optimized for real-time performance.

- 📚 **Educational Mode**  
  Learn ASL with interactive tutorials and practice exercises.

---

## 🖼️ Screenshots

![image](https://github.com/user-attachments/assets/60d936a2-0503-429d-bb2b-7fb49cf45bab)

![image](https://github.com/user-attachments/assets/252b6bd1-cf7d-48b8-abf8-046546eb4317)


---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.10 (⚠️ Avoid Python 3.13+)
- A webcam
- Internet connection (for first-time setup)

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/KAsHiSHSET/Sign_speak.git
cd SignSpeak

# Create and activate a virtual environment
python -m venv venv
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
