# 🤖 AI_GYM – AI-Powered Fitness Coach  

**AI_GYM** is a high-performance, AI-driven fitness ecosystem that delivers **hyper-personalized coaching, real-time form correction, and holistic progress tracking**. It leverages computer vision (MediaPipe & OpenCV) and generative deep learning (Decoder-only Transformer) to transform raw biometric metrics into tailored training plans and safe workouts.

🔗 **Repository:** [AI_GYM](https://github.com/AlysterBenedict/AI_GYM.git)

---

## 🚀 Project Vision & Core Features

### 1. Intelligent Form Correction & Guidance
* **Real-Time Pose Estimation:** BlazePose GHUM 3D tracks 33 standard human keypoints.
* **Core Exercise Engine:** Audits angles and movement ranges to track reps and holds for 50 distinct exercises.
* **Live Form Feedback:** Delivers visual and auditory cues (e.g., "Go Deeper", "Bend Over More") with dynamic HUD color states (Green for correct form, Red for posture warnings).

### 2. Autoregressive Workout Generation
* **Biometric Optimization:** Translates age, gender, level, BMI, chest, waist, hips, thighs, and biceps into customized programs.
* **Deep Learning Synthesis:** Employs a custom Decoder-Only Transformer to sequence daily workouts progressive over a 30-day timeline.
* **Smart Training Rules:** Programmatically builds splits (Push/Pull/Legs, HIIT, Cardio) and schedules progressive overload and a deload cycle (Week 3).

### 3. Integrated Microservices Backend
* **REST API:** Production FastAPI service hosted via Uvicorn to parse images and user biometrics.
* **Client Synchronization:** Standardizes and exports key preprocessing parameters to JSON format for offline Android/Kotlin execution.

---

## 📂 Deep Dive: Repository Structure & File Registry

The workspace is organized into modular components reflecting distinct stages of model development, validation, and production hosting.

```
AI_GYM/
│── .gradio/                           # Gradio interface configuration backups
│── .vscode/                           # VS Code workspace settings
│── bodym_data/                        # Workspace folder containing raw biometric dataset images
│── details/                           # Additional metadata, reports, and scripts
│── docs/                              # Project scope and research documentation
│   ├── BodyM Dataset Details.pdf      # Detailed parameters for the BodyM biometric dataset
│   ├── Dataaset finding.pdf           # Analysis of target datasets for human body metric estimation
│   ├── Model Card BlazePose GHUM 3D.pdf # MediaPipe BlazePose keypoint estimation model card
│   ├── project scope.docx             # Project vision, requirements, and development roadmap
│   ├── report format 1.pdf            # Analysis templates and visual metrics formatting guidelines
│   └── report format 2.pdf            # Research report template layout
│
│── Body Metrics Model Training/       # Front/Side Biometric Regression Training
│   └── body_model_training.ipynb      # Notebook detailing the training of the biometric estimation model
│
│── Pipeline Testing/                  # End-to-End Evaluation Environments
│   ├── image_pipeline_testing.ipynb   # Visual inspection playground (Original -> Background Removal -> Silhouette)
│   └── pipeline_testing_app.py        # Gradio pipeline app loading Vision Estimator + Transformer + LLM Coach Chat
│
│── Pose estimation/                   # Live Computer Vision & Exercise Auditing
│   ├── pose50.ipynb                   # Real-time form estimation and keybind sequential loops for 50 exercises
│   └── pose_sample.ipynb              # Baseline real-time pose estimation playground with 10 exercises
│
│── Preprocessing Json/                # Client-Side Synchronization & Cross-Platform Utilities
│   ├── generate_preprocessing_data.py # Utility script to export StandardScaler and MinMaxScaler properties to JSON
│   └── preprocessing_data.json        # Pretrained scaling factors for client apps (Kotlin/Java)
│
│── Workout generator training/        # Sequence Transformer Model Development
│   ├── generate_training_dataset.py   # Synthetic data generator program producing 100k progressive 30-day profiles
│   ├── train_workout_generator.py     # Causal decoder-only Transformer training script (using AMP & Gradient Accumulation)
│   ├── manual_workout_plan_ui.py      # Standalone text-input Gradio UI for quick manual transformer testing
│   ├── my_30_day_workout_plan.csv     # Local sample generated plan output
│   └── trainium_production_dataset_100k.csv # Ingested 100k training dataset CSV
│
│── fast_api_app/                      # Production REST API Backend
│   ├── recommendation/                # API Recommendation logic packages
│   │   ├── biometric_estimator.py     # Wraps dual-branch EfficientNet and rembg/CV2 silhouette preprocessing
│   │   ├── workout_generator.py       # Autoregressive Decoder-Only Transformer wrapper with Top-K (k=10) sampling
│   │   └── llm_explainer.py           # Integrates local LM Studio google/gemma-4-26b-a4b narrative generation for plans
│   │
│   ├── best_bodym_model.pth           # Neural network weights for silhouette metric estimation
│   ├── trainium_sota_transformer_model.pth # PyTorch weights for workout plan generator
│   ├── scaler.pkl                     # MinMaxScaler attributes for numerical biometrics
│   ├── encoder.pkl                    # LabelEncoder categories for categorical biometrics
│   ├── tokenizer.json                 # Vocabulary dictionary mapping exercise IDs to names
│   ├── requirements.txt               # API-specific dependencies
│   └── main.py                        # Uvicorn entrypoint defining FastAPI routes (/predict_biometrics, /generate_workout)
│
│── requirements.txt                   # Universal training/testing dependencies
│── Trainium.apk                       # Compiled Android build package
│── .gitattributes
│── .gitignore
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlysterBenedict/AI_GYM.git
   cd AI_GYM
   ```

2. **Create the Conda Environment:**
   Create an isolated environment named `metrics` built on Python 3.10:
   ```bash
   conda create -n metrics python=3.10 -y
   ```

3. **Activate the Environment:**
   ```bash
   conda activate metrics
   ```

4. **Install Universal Dependencies:**
   Install the exact, cross-referenced versions of ML, Vision, and Server packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Running the Production FastAPI Server

The `fast_api_app` hosts the deep learning inference engines via high-performance REST APIs. It runs isolated on your local network to feed biometric estimation and progressive plans to external frontend clients.

### Method to Run the Backend:

Open your PowerShell terminal and execute the following commands in sequence:

1. **Load your shell profile (ensure environment variables are configured):**
   ```powershell
   .$PROFILE
   ```

2. **Navigate into the FastAPI project directory:**
   ```powershell
   cd "C:\Users\bened\Documents\Alyster Coding\PROJECTS\AI_GYM\fast_api_app"
   ```

3. **Activate the conda environment containing verified CUDA, PyTorch, and MediaPipe dependencies:**
   ```powershell
   conda activate metrics
   ```

4. **Start the high-performance ASGI server with live-reloads enabled:**
   ```powershell
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

Once initialized, the interactive documentation is accessible at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc`.

---

## 📜 Technical Acknowledgements

* **MediaPipe BlazePose:** For high-fidelity real-time 33 human skeletal keypoint predictions.
* **BodyM Dataset:** CC BY-NC 4.0 – For providing baseline anthropometric coordinates for training the EfficientNet regression branch.
* **PyTorch & torchvision:** Powers the mixed-precision training operations and inference loops.

---
With **AI_GYM**, fitness meets intelligence. Stay consistent, stay motivated, and let AI guide your journey.
