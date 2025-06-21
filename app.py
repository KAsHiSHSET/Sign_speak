from flask import Flask, render_template, Response, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import warnings
import os
import threading
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Global variables for processing
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ', 37: '.'
}

expected_features = 42
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
last_registered_time = time.time()
registration_delay = 1.5
is_paused = False
current_alphabet = "N/A"
expected_features = 42

# Function to generate camera frames
def generate_frames():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, is_paused, current_alphabet
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if not is_paused:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Ensure valid data
                    if len(data_aux) < expected_features:
                        data_aux.extend([0] * (expected_features - len(data_aux)))
                    elif len(data_aux) > expected_features:
                        data_aux = data_aux[:expected_features]

                    # Predict gesture
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Stabilization logic
                    stabilization_buffer.append(predicted_character)
                    if len(stabilization_buffer) > 30:  # Buffer size for 1 second
                        stabilization_buffer.pop(0)

                    if stabilization_buffer.count(predicted_character) > 25:  # Stabilization threshold
                        # Register the character only if enough time has passed since the last registration
                        current_time = time.time()
                        if current_time - last_registered_time > registration_delay:
                            stable_char = predicted_character
                            last_registered_time = current_time  # Update last registered time
                            current_alphabet = stable_char

                            # Handle word and sentence formation
                            if stable_char == ' ':
                                if word_buffer.strip():  # Add word to sentence if not empty
                                    sentence += word_buffer + " "
                                word_buffer = ""
                            elif stable_char == '.':
                                if word_buffer.strip():  # Add word to sentence before adding period
                                    sentence += word_buffer + "."
                                word_buffer = ""
                            else:
                                word_buffer += stable_char

                    # Draw landmarks and bounding box
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())

        # Draw alphabet on the video feed
        cv2.putText(frame, f"Alphabet: {current_alphabet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global current_alphabet, word_buffer, sentence
    return jsonify({
        'current_alphabet': current_alphabet,
        'current_word': word_buffer if word_buffer else "N/A",
        'current_sentence': sentence if sentence else "N/A",
        'is_paused': is_paused
    })

@app.route('/reset_sentence')
def reset_sentence():
    global word_buffer, sentence, current_alphabet
    word_buffer = ""
    sentence = ""
    current_alphabet = "N/A"
    return jsonify({'status': 'success'})

@app.route('/toggle_pause')
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'status': 'success', 'is_paused': is_paused})

@app.route('/speak_sentence')
def speak_sentence():
    global sentence
    # The actual speaking will be handled by JavaScript in the frontend
    return jsonify({'status': 'success', 'sentence': sentence})

if __name__ == '__main__':
    app.run(debug=True)