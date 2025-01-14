import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from threading import Thread
import pyttsx3
from datetime import datetime
import string
from PIL import Image, ImageTk
import random
import time

class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.signs = {i: letter for i, letter in enumerate(string.ascii_uppercase)}
        self.data = []
        self.labels = []
        self.model = None
        self.cap = None
        self.running = False
        self.engine = pyttsx3.init()
        self.last_spoken = None
        self.current_samples = 0
        self.total_samples = 0
        
        # Test mode attributes
        self.test_running = False
        self.current_test_letter = None
        self.test_start_time = None
        self.correct_answers = 0
        self.total_questions = 0
        self.test_letters = None

    def collect_data(self, callback, samples_per_sign=50):
        """Collect sign language data for training."""
        self.total_samples = samples_per_sign * len(self.signs)
        self.current_samples = 0
        
        if not os.path.exists('sign_data'):
            os.makedirs('sign_data')
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Cannot access camera. Please check your camera connection.")

        for sign_num, sign in self.signs.items():
            samples_collected = 0
            
            instruction_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(instruction_frame, f"Ready to collect sign: {sign}", 
                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(instruction_frame, "Press SPACE to start", 
                       (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Data Collection', instruction_frame)
            
            while True:
                if cv2.waitKey(1) & 0xFF == 32:  # SPACE
                    break
            
            while samples_collected < samples_per_sign:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.data.append(landmarks)
                    self.labels.append(sign_num)
                    samples_collected += 1
                    self.current_samples += 1
                    
                    progress = (self.current_samples / self.total_samples) * 100
                    callback(progress, sign)
                
                cv2.putText(frame, f"Sign: {sign}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_sign}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            np.save('sign_data/data.npy', np.array(self.data))
            np.save('sign_data/labels.npy', np.array(self.labels))
        
        self.cap.release()
        cv2.destroyAllWindows()

    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks."""
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks = landmarks / max_dist
        return landmarks.flatten()

    def train_model(self, callback):
        """Train the sign recognition model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        try:
            X = np.load('sign_data/data.npy')
            y = np.load('sign_data/labels.npy')
        except FileNotFoundError:
            raise Exception("Training data not found. Please collect data first.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        callback(25, "Training model...")
        
        self.model.fit(X_train, y_train)
        callback(75, "Evaluating model...")
        
        accuracy = self.model.score(X_test, y_test)
        callback(90, f"Accuracy: {accuracy:.2f}")
        
        with open('sign_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        callback(100, f"Training complete! Accuracy: {accuracy:.2f}")

    def predict_sign(self, frame, results):
        """Predict the sign from a single frame."""
        if not results.multi_hand_landmarks:
            return None, 0
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = self.extract_landmarks(hand_landmarks)
        X = landmarks.reshape(1, -1)
        
        prediction = self.model.predict(X)[0]
        confidence = np.max(self.model.predict_proba(X)[0])
        
        return self.signs[prediction], confidence * 100

    def start_prediction(self, frame_callback, prediction_callback):
        """Start real-time prediction."""
        try:
            with open('sign_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")
        
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                prediction, confidence = self.predict_sign(frame, results)
                
                if prediction and confidence > 70:
                    cv2.putText(frame, f"{prediction} ({confidence:.1f}%)", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    prediction_callback(prediction, confidence)
                    
                    if prediction != self.last_spoken:
                        self.engine.say(prediction)
                        self.engine.runAndWait()
                        self.last_spoken = prediction
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_callback(frame_rgb)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def start_test_mode(self, frame_callback, score_callback, letter_callback, custom_letters=None):
        """Start the testing mode with optional custom letters."""
        try:
            with open('sign_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")
        
        self.cap = cv2.VideoCapture(0)
        self.test_running = True
        self.correct_answers = 0
        self.total_questions = 0
        self.test_letters = custom_letters if custom_letters else list(string.ascii_uppercase)
        
        while self.test_running:
            if self.current_test_letter is None or \
               time.time() - self.test_start_time > 15:  # 15 second timer
                # Generate new letter from available set
                self.current_test_letter = random.choice(self.test_letters)
                self.test_start_time = time.time()
                letter_callback(self.current_test_letter)
                self.total_questions += 1
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame,  1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw timer with color gradient
            remaining_time = 15 - (time.time() - self.test_start_time)
            time_color = (
                int(255 * (1 - remaining_time/15)),  # Red increases as time runs out
                int(255 * (remaining_time/15)),      # Green decreases as time runs out
                0
            )
            cv2.putText(frame, f"Time: {int(remaining_time)}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                prediction, confidence = self.predict_sign(frame, results)
                
                if prediction and confidence > 70:
                    if prediction == self.current_test_letter:
                        self.correct_answers += 1
                        # Generate new letter immediately on correct answer
                        self.current_test_letter = None
                    
                    cv2.putText(frame, f"Your sign: {prediction}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update score
            score = (self.correct_answers / self.total_questions * 100) if self.total_questions > 0 else 0
            score_callback(score, self.correct_answers, self.total_questions)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_callback(frame_rgb)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_prediction(self):
        """Stop the prediction process."""
        self.running = False
        self.test_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SignMentor")
        self.root.geometry("1200x800")
        
        # Initialize detector
        self.detector = SignLanguageDetector()
        
        # Setup theme
        self.setup_theme()
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize variables
        self.prediction_history = []
        self.custom_letters = None

    def setup_theme(self):
        """Configure the application theme and styles."""
        self.root.configure(bg='#f0f2f5')
        style = ttk.Style()
        
        # Configure custom styles
        style.configure("Custom.TFrame", background='#f0f2f5')
        style.configure("Card.TFrame", background='#ffffff', relief='solid')
        style.configure("Custom.TButton", 
                       padding=10, 
                       font=('Helvetica', 10, 'bold'),
                       background='#4a90e2')
        style.configure("Custom.TLabel", 
                       background='#f0f2f5', 
                       font=('Helvetica', 10))
        style.configure("Title.TLabel", 
                       background='#f0f2f5', 
                       font=('Helvetica', 20, 'bold'),
                       foreground='#2c3e50')
        style.configure("Prediction.TLabel", 
                       background='#ffffff', 
                       font=('Helvetica', 24, 'bold'),
                       foreground='#4a90e2')
        
        # Configure progress bar style
        style.configure("Custom.Horizontal.TProgressbar",
                       troughcolor='#f0f2f5',
                       background='#4a90e2',
                       thickness=20)

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        self.main_container = ttk.Frame(self.root, style="Custom.TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel (Controls)
        self.left_panel = ttk.Frame(self.main_container, style="Custom.TFrame")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(self.left_panel, 
                               text="Sign Language Learning Assistant", 
                               style="Title.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Control buttons
        self.create_control_buttons()
        
        # Progress section
        self.create_progress_section()
        
        # Right panel (Video feed and predictions)
        self.right_panel = ttk.Frame(self.main_container, style="Custom.TFrame")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video feed
        self.video_label = ttk.Label(self.right_panel)
        self.video_label.pack(pady=10)
        
        # Predictions display
        self.create_predictions_display()
    
    def create_control_buttons(self):
        """Create all control buttons."""
        # Buttons frame
        buttons_frame = ttk.Frame(self.left_panel, style="Card.TFrame")
        buttons_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Control buttons
        self.collect_button = ttk.Button(buttons_frame, text="Collect Data", 
                                       command=self.collect_data, style="Custom.TButton")
        self.collect_button.pack(fill=tk.X, pady=5, padx=5)
        
        self.train_button = ttk.Button(buttons_frame, text="Train Model", 
                                     command=self.train_model, style="Custom.TButton")
        self.train_button.pack(fill=tk.X, pady=5, padx=5)
        
        self.predict_button = ttk.Button(buttons_frame, text="Start Detection", 
                                       command=self.start_prediction, style="Custom.TButton")
        self.predict_button.pack(fill=tk.X, pady=5, padx=5)
        
        self.test_button = ttk.Button(buttons_frame, text="Start Test Mode", 
                                    command=self.start_test_mode, style="Custom.TButton")
        self.test_button.pack(fill=tk.X, pady=5, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", 
                                    command=self.stop_prediction, state="disabled",
                                    style="Custom.TButton")
        self.stop_button.pack(fill=tk.X, pady=5, padx=5)
        
        # Save/Load model buttons
        self.save_button = ttk.Button(buttons_frame, text="Save Model", 
                                    command=self.save_model, style="Custom.TButton")
        self.save_button.pack(fill=tk.X, pady=5, padx=5)
        
        self.load_button = ttk.Button(buttons_frame, text="Load Model", 
                                    command=self.load_model, style="Custom.TButton")
        self.load_button.pack(fill=tk.X, pady=5, padx=5)
    
    def create_progress_section(self):
        """Create the progress section."""
        # Progress frame
        progress_frame = ttk.Frame(self.left_panel, style="Card.TFrame")
        progress_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Status: Ready", 
                                    style="Custom.TLabel", wraplength=200)
        self.status_label.pack(fill=tk.X, pady=5, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          mode='determinate', 
                                          style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5, padx=5)
    
    def create_predictions_display(self):
        """Create the predictions and test display sections."""
        # Predictions frame
        predictions_frame = ttk.Frame(self.right_panel, style="Card.TFrame")
        predictions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Current prediction
        self.prediction_label = ttk.Label(predictions_frame, text="Prediction: None", 
                                        style="Prediction.TLabel")
        self.prediction_label.pack(pady=10)
        
        # Confidence meter
        self.confidence_var = tk.DoubleVar()
        self.confidence_meter = ttk.Progressbar(predictions_frame, 
                                              variable=self.confidence_var,
                                              mode='determinate',
                                              style="Custom.Horizontal.TProgressbar")
        self.confidence_meter.pack(fill=tk.X, padx=20, pady=5)
        
        # Test mode frame
        self.test_frame = ttk.Frame(predictions_frame, style="Card.TFrame")
        self.test_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Current letter to sign
        self.test_letter_label = ttk.Label(self.test_frame, 
                                         text="Sign this letter: ", 
                                         style="Prediction.TLabel")
        self.test_letter_label.pack(pady=10)
        
        # Score display
        self.score_label = ttk.Label(self.test_frame, 
                                   text="Score: 0% (0/0)", 
                                   style="Custom.TLabel")
        self.score_label.pack(pady=5)
        
        # History frame
        history_frame = ttk.Frame(predictions_frame, style="Card.TFrame")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        history_label = ttk.Label(history_frame, text="Recent Predictions:", 
                                style="Custom.TLabel")
        history_label.pack(anchor=tk.W, padx=5)
        
        self.history_text = tk.Text(history_frame, height=5, width=40)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_custom_test_dialog(self):
        """Create a dialog for custom letter selection."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Custom Test Settings")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create and pack widgets
        ttk.Label(dialog, 
                 text="Select letters to practice:", 
                 style="Title.TLabel").pack(pady=10)
        
        # Create checkbuttons for letters
        letter_var_dict = {}
        letter_frame = ttk.Frame(dialog, style="Card.TFrame")
        letter_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        for i, letter in enumerate(string.ascii_uppercase):
            var = tk.BooleanVar()
            letter_var_dict[letter] = var
            cb = ttk.Checkbutton(letter_frame, text=letter, variable=var)
            cb.grid(row=i//6, column=i%6, padx=5, pady=5)
        
        def start_custom_test():
            selected_letters = [l for l, v in letter_var_dict.items() if v.get()]
            if not selected_letters:
                messagebox.showwarning("Warning", "Please select at least one letter!")
                return
            self.custom_letters = selected_letters
            dialog.destroy()
            self.start_test_mode()
        
        ttk.Button(dialog, 
                  text="Start Custom Test", 
                  command=start_custom_test,
                  style="Custom.TButton").pack(pady=20)

    def show_test_results(self):
        """Show a summary of test results."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Test Results")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        score = (self.detector.correct_answers / self.detector.total_questions * 100) \
                if self.detector.total_questions > 0 else 0
        
        # Create results display
        results_frame = ttk.Frame(dialog, style="Card.TFrame")
        results_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        ttk.Label(results_frame, 
                 text="Test Complete!", 
                 style="Title.TLabel").pack(pady=10)
        
        ttk.Label(results_frame,
                 text=f"Final Score: {score:.1f}%",
                 style="Prediction.TLabel").pack(pady=10)
        
        ttk.Label(results_frame,
                 text=f"Correct Answers: {self.detector.correct_answers}",
                 style="Custom.TLabel").pack(pady=5)
        
        ttk.Label(results_frame,
                 text=f"Total Questions: {self.detector.total_questions}",
                 style="Custom.TLabel").pack(pady=5)
        
        ttk.Button(dialog,
                  text="Close",
                  command=dialog.destroy,
                  style="Custom.TButton").pack(pady=20)
    
    def update_frame(self, frame):
        """Update the video feed display."""
        frame = cv2.resize(frame, (640, 480))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo)
    
    def update_prediction(self, prediction, confidence):
        """Update the prediction display and history."""
        self.prediction_label.configure(text=f"Prediction: {prediction}")
        self.confidence_var.set(confidence)
        
        # Update history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.prediction_history.append(f"{timestamp} - {prediction} ({confidence:.1f}%)")
        
        # Keep only last 10 predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Update history display
        self.history_text.delete(1.0, tk.END)
        for pred in reversed(self.prediction_history):
            self.history_text.insert(tk.END, pred + "\n")
    
    def update_test_score(self, score, correct, total):
        """Update the test score display."""
        self.score_label.configure(text=f"Score: {score:.1f}% ({correct}/{total})")
    
    def update_test_letter(self, letter):
        """Update the display of the current letter to sign."""
        self.test_letter_label.configure(text=f"Sign this letter: {letter}")
    
    def collect_data(self):
        """Start data collection process."""
        self.update_button_states("collecting")
        
        def progress_callback(progress, current_sign):
            self.progress_var.set(progress)
            self.status_label.configure (text=f"Collecting data for sign: {current_sign}")
        
        try:
            Thread(target=lambda: self.detector.collect_data(progress_callback), 
                  daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_button_states("ready")
    
    def train_model(self):
        """Start model training process."""
        self.update_button_states("training")
        
        def progress_callback(progress, status):
            self.progress_var.set(progress)
            self.status_label.configure(text=status)
            
            if progress == 100:
                self.update_button_states("ready")
                messagebox.showinfo("Training Complete", status)
        
        try:
            Thread(target=lambda: self.detector.train_model(progress_callback), 
                  daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_button_states("ready")
    
    def start_prediction(self):
        """Start real-time prediction."""
        try:
            self.update_button_states("predicting")
            Thread(target=lambda: self.detector.start_prediction(
                self.update_frame, self.update_prediction), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_button_states("ready")
    
    def start_test_mode(self):
        """Show dialog to choose between regular and custom test."""
        if not hasattr(self, 'test_dialog'):
            self.test_dialog = tk.Toplevel(self.root)
            self.test_dialog.title("Choose Test Mode")
            self.test_dialog.geometry("300x200")
            self.test_dialog.transient(self.root)
            self.test_dialog.grab_set()
            
            ttk.Label(self.test_dialog,
                     text="Select Test Mode",
                     style="Title.TLabel").pack(pady=20)
            
            ttk.Button(self.test_dialog,
                      text="Full Alphabet Test",
                      command=self.start_full_test,
                      style="Custom.TButton").pack(pady=10)
            
            ttk.Button(self.test_dialog,
                      text="Custom Letters Test",
                      command=self.start_custom_test,
                      style="Custom.TButton").pack(pady=10)
    
    def start_full_test(self):
        """Start test with full alphabet."""
        self.test_dialog.destroy()
        delattr(self, 'test_dialog')
        self.custom_letters = None
        try:
            self.update_button_states("testing")
            Thread(target=lambda: self.detector.start_test_mode(
                self.update_frame,
                self.update_test_score,
                self.update_test_letter,
                self.custom_letters), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_button_states("ready")
    
    def start_custom_test(self):
        """Start test with custom letters."""
        self.test_dialog.destroy()
        delattr(self, 'test_dialog')
        self.create_custom_test_dialog()
    
    def stop_prediction(self):
        """Stop the prediction process."""
        self.detector.stop_prediction()
        self.update_button_states("ready")
        if hasattr(self, 'test_dialog'):
            self.test_dialog.destroy()
            delattr(self, 'test_dialog')
            self.show_test_results()
    
    def save_model(self):
        """Save the trained model to a file."""
        if not self.detector.model:
            messagebox.showerror("Error", "No trained model available to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.detector.model, f)
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load a trained model from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    self.detector.model = pickle.load(f)
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def update_button_states(self, state):
        """Update the state of buttons based on the current operation."""
        if state == "collecting":
            self.collect_button.config(state="disabled")
            self.train_button.config(state="disabled")
            self.predict_button.config(state="disabled")
            self.test_button.config(state="disabled")
            self.stop_button.config(state="normal")
        elif state == " training":
            self.collect_button.config(state="disabled")
            self.train_button.config(state="disabled")
            self.predict_button.config(state="disabled")
            self.test_button.config(state="disabled")
            self.stop_button.config(state="disabled")
        elif state == "predicting":
            self.collect_button.config(state="disabled")
            self.train_button.config(state="disabled")
            self.predict_button.config(state="disabled")
            self.test_button.config(state="disabled")
            self.stop_button.config(state="normal")
        elif state == "testing":
            self.collect_button.config(state="disabled")
            self.train_button.config(state="disabled")
            self.predict_button.config(state="disabled")
            self.test_button.config(state="disabled")
            self.stop_button.config(state="normal")
        else:  # ready state
            self.collect_button.config(state="normal")
            self.train_button.config(state="normal")
            self.predict_button.config(state="normal")
            self.test_button.config(state="normal")
            self.stop_button.config(state="disabled") 
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()