# Sign Language Learning Assistant

A machine learning-based application designed to help users learn and practice sign language through real-time detection and feedback.

<img width="893" alt="image" src="https://github.com/user-attachments/assets/f084fa8e-f65b-4c9d-883c-012489cba223" />

## 🌟 Features

- Real-time sign language detection and recognition
- Interactive learning mode
- Practice mode with scoring system
- Model training capabilities
- Data collection tools
- Save and load trained models
- Performance testing mode
- Text-to-speech feedback
- GUI interface built with Tkinter

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Webcam or camera device

### Required Dependencies

```bash
pip install -r requirements.txt
```

Dependencies list (`requirements.txt`):
```
opencv-python       # Computer vision and image processing
mediapipe          # Hand and pose detection framework
numpy              # Numerical computing and array operations
pickle             # Model serialization
tkinter            # GUI framework (usually comes with Python)
pyttsx3           # Text-to-speech conversion
Pillow            # Image processing library
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sign-language-assistant.git
cd sign-language-assistant
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📱 Usage

The application offers several key functionalities through its intuitive interface:

### Data Collection
- Click "Collect Data" to start gathering training data
- Follow the on-screen instructions to record signs
- Ensure proper lighting and camera positioning

### Training
- Use "Train Model" to begin the training process
- Monitor the training progress in the status window
- Wait for completion notification

### Detection
- Click "Start Detection" to begin real-time sign language recognition
- Position yourself in front of the camera
- Perform signs to see instant recognition results

  

https://github.com/user-attachments/assets/12d0fecb-4844-4316-b836-7b20694adfbe



### Testing
- Use "Start Test Mode" to evaluate your sign language skills
- Follow the prompts to perform specific signs
- Receive immediate feedback and scoring

  

https://github.com/user-attachments/assets/a10df58f-e555-4bde-a671-09ab89ff5a58



### Model Management
- Save your trained models using "Save Model"
- Load previously trained models using "Load Model"
- Stop the current session using "Stop"

## 📊 Performance Metrics

- Real-time prediction display
- Accuracy scoring system
- Recent predictions history
- Status monitoring

## 🛠️ Technical Details

The application utilizes several key technologies:
- OpenCV (cv2): Handles video capture and image processing
- MediaPipe: Provides hand tracking and gesture recognition
- NumPy: Manages numerical operations and data processing
- Tkinter: Creates the graphical user interface
- pyttsx3: Enables text-to-speech feedback
- Threading: Ensures smooth GUI operation during processing
- Pickle: Handles model serialization and deserialization
- PIL (Pillow): Processes images for the GUI display

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📮 Contact

For questions, feedback, or support, please open an issue in the GitHub repository.
