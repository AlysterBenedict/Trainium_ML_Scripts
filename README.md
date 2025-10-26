
---

#  AI_GYM – AI-Powered Fitness Coach  

**AI_GYM** is an AI-driven fitness ecosystem that delivers **hyper-personalized coaching, real-time form correction, and holistic progress tracking**. Built for Android, it transforms your device into an intelligent fitness companion that motivates, guides, and ensures safe, effective workouts.  

🔗 **Repository:** [AI_GYM](https://github.com/AlysterBenedict/AI_GYM.git)  

---

## Project Vision  
To create an intelligent fitness companion that:  
- Provides **real-time form correction** using computer vision.  
- Generates **personalized workout & nutrition plans**.  
- Tracks **biometrics, progress, and consistency**.  
- Motivates users with **AI-powered coaching & support**.  

---

##  Core Features  

###  Intelligent Form Correction & Guidance  
- **Real-Time Pose Estimation** via BlazePose GHUM 3D.  
- **AI-Powered Form Analysis** with 33 body keypoints.  
- **Instant Feedback** through audio cues & visual overlays.  

###  Hyper-Personalized Training & Nutrition  
- AI-driven **workout generation** based on biometrics & body type.  
- **Goal alignment** for weight loss, muscle gain, or healthy mass gain.  
- **Diet & nutrition guidance** with macronutrient breakdowns.  

###  Advanced Progress & Biometric Tracking  
- **AI weight & progress tracker** with motivational feedback.  
- **Visual progress calendar** for streaks & consistency.    
- **Dynamic dashboards** with charts & insights.
- **Wear OS integration** for heart rate, calories, and workout duration.


###  AI-Powered Motivation & Support  
- **24/7 conversational AI coach** for tips & guidance.  
- **Proactive engagement** with reminders & motivational nudges.  

---

##  Tech Stack  

- **Frontend:** Android (Kotlin)  
- **Backend:** Firebase (authentication, cloud storage, real-time DB)  
- **AI Models:**  
  - [BodyM Dataset](https://registry.opendata.aws/bodym/) for biometric extraction.  
  - [BlazePose GHUM 3D](https://google.github.io/mediapipe/solutions/pose.html) for real-time pose estimation.  
- **Deployment:** Google Play Store/ Local apk download.

---

## 📂 Project Structure  
```
AI_GYM/
│── .gradio/                     # Gradio UI configs
│── .vscode/                     # VS Code workspace settings
│── datasets/                    # Training and evaluation datasets
│── docs/                        # Documentation, research notes, model cards
|
│── .gitattributes
│── .gitignore
│── best_bodym_model.pth         # Trained BodyM model weights
│── body_model_training.ipynb    # Notebook for BodyM model training
│── encoder.pkl                  # Saved encoder for preprocessing
│── image_pipeline.ipynb         # Image preprocessing & pipeline notebook
│── pose_sample.ipynb            # Sample notebook for pose estimation
│── pose50.ipynb                 # Extended pose estimation experiments for 50 exercises
│── README.md                    # Project overview
│── requirements.txt             # Python dependencies
│── scaler.pkl                   # Feature scaler for normalization
│── tokenizer.json               # Tokenizer for text/labels
│── train_sota_transformer_model.py  # Training script for transformer model
│── trainium_production_dataset_100k.csv  # Production dataset
│── trainium_sota_transformer_model.pth   # Transformer model weights
│── Trainium.apk                 # Android build 
│── workout_plan_csv.py          # Script to generate workout plans (CSV)
│── workout_plan_ui.py           # Script for workout plan gradio UI
```
---

##  Installation & Setup  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AlysterBenedict/AI_GYM.git
   cd AI_GYM
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run the app on an android device**  
   ```bash
   Trainium.apk
   ```


---

##  Roadmap  

- [ ] Expand **nutrition database** with regional foods.  
- [ ] Integrate **voice-based coaching**.  
 
---


---

## 📜 License  

This project is licensed under the **Apache 2.0 License**.  
Datasets used:  
- **BodyM Dataset** – CC BY-NC 4.0  
- **BlazePose GHUM 3D** – Apache 2.0  

---

##  Acknowledgements  

- [Google MediaPipe BlazePose](https://google.github.io/mediapipe/)  
- [BodyM Dataset](https://registry.opendata.aws/bodym/)  
 

---

With **AI_GYM**, fitness meets intelligence. Stay consistent, stay motivated, and let AI guide your journey!  

---


