# 🩻 XRayAI — AI-Powered Chest X-Ray Analysis Platform

> **An AI-powered medical imaging platform that analyzes chest X-rays, classifies findings, and generates structured diagnostic reports through a modern web application.**

XRayAI is a full-stack **AI-powered chest X-ray analysis system** designed to demonstrate how modern deep learning, medical imaging, backend APIs, and intelligent reporting can be combined into a practical healthcare application.

The system uses a **medical-domain pretrained DenseNet121 model from TorchXRayVision**, fine-tuned for three-class chest X-ray classification:

* 🦠 **COVID-19**
* 🫁 **Pneumonia**
* ✅ **Normal**

The project combines AI inference with a complete web platform featuring authentication, analysis history, PDF report generation, and an integrated AI assistant.

---

## 🚀 Key Highlights

* 🧠 **Medical-domain pretrained DenseNet121**
* 🎯 **91.98% test accuracy** on the project's held-out test set
* 🩻 Automated chest X-ray classification
* 🦠 COVID-19 detection
* 🫁 Pneumonia detection
* ✅ Normal X-ray classification
* ⚡ FastAPI backend
* 🌐 Modern web frontend
* 🔐 User authentication and registration
* 📊 Analysis history
* 📄 Automated PDF report generation
* 🤖 Integrated AI chatbot/assistant
* 🗄️ Database-backed application
* 🎨 Light/Dark theme support
* 🧪 Model testing and evaluation scripts
* 🏗️ Modular backend architecture

---

# 🧠 AI Model

XRayAI uses a **DenseNet121 architecture pretrained on medical chest X-ray data through TorchXRayVision**.

Instead of starting from a generic computer-vision model, the project uses a model with prior exposure to medical X-ray representations and then adapts it for the project's three target classes.

### Classification Pipeline

```text
                 Chest X-Ray
                      │
                      ▼
             Image Preprocessing
                      │
                      ▼
       Medical Pretrained DenseNet121
                      │
                      ▼
             Feature Extraction
                      │
                      ▼
             Classification Head
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        COVID       NORMAL     PNEUMONIA
```

### Classification Head

The pretrained backbone is followed by a custom classification head:

```text
DenseNet121
    │
    ▼
1024 Features
    │
    ▼
Linear Layer
1024 → 512
    │
    ▼
ReLU
    │
    ▼
Dropout (0.4)
    │
    ▼
Linear Layer
512 → 3
    │
    ▼
COVID / NORMAL / PNEUMONIA
```

The medical pretrained backbone is used as the foundation while the classification head is adapted for the project's target classes.

---

# 📊 Model Performance

The final trained model was evaluated on a held-out test set containing:

| Class     | Test Images |
| --------- | ----------: |
| COVID     |         542 |
| NORMAL    |         509 |
| PNEUMONIA |         583 |
| **Total** |   **1,634** |

### Overall Performance

**Test Accuracy: 91.98%**

### Classification Performance

| Class     | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| COVID     |    93.05% | 91.33% |   92.18% |
| NORMAL    |    90.41% | 90.77% |   90.59% |
| PNEUMONIA |    92.39% | 93.65% |   93.02% |

> These results are from the project's test evaluation and should not be interpreted as clinical validation or evidence of suitability for medical diagnosis.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│                  FRONTEND                   │
│                                             │
│  Login │ Register │ X-Ray Upload │ History │
│        │ Profile  │ Reports      │ Chatbot │
└──────────────────────┬──────────────────────┘
                       │
                       │ HTTP / REST API
                       ▼
┌─────────────────────────────────────────────┐
│                FASTAPI BACKEND              │
│                                             │
│ Authentication                              │
│ X-Ray Processing                            │
│ AI Inference                                │
│ Database Operations                         │
│ PDF Report Generation                       │
│ AI Assistant Integration                    │
└──────────────────────┬──────────────────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
┌─────────────────────┐  ┌───────────────────┐
│   AI MODEL          │  │    DATABASE       │
│                     │  │                   │
│ DenseNet121         │  │ Users             │
│ TorchXRayVision     │  │ Analysis History  │
│ Classification      │  │ Results           │
└─────────────────────┘  └───────────────────┘
             │
             ▼
      ┌───────────────┐
      │ PDF REPORT    │
      │ GENERATION    │
      └───────────────┘
```

---

# 💻 Technology Stack

### AI / Machine Learning

* Python
* PyTorch
* TorchXRayVision
* DenseNet121
* Deep Learning
* Transfer Learning
* Medical Image Classification

### Backend

* FastAPI
* Python
* REST APIs
* Database integration
* ReportLab

### Frontend

* HTML5
* CSS3
* JavaScript
* Responsive UI
* Light/Dark Theme

### Database

* SQL-based persistence
* User management
* Analysis history
* Prediction records

### Development

* Git
* GitHub
* Virtual Environment
* VS Code

---

# 📁 Project Structure

```text
XRayAI/
│
├── backend/
│   ├── database.py
│   ├── main.py
│   ├── medsiglip_test.py
│   ├── models.py
│   ├── pdf_generator.py
│   └── test_gemini_xray.py
│
├── frontend/
│   ├── auth.js
│   ├── chatbot.js
│   ├── history.html
│   ├── history.js
│   ├── index.html
│   ├── login.html
│   ├── profile.html
│   ├── profile.js
│   ├── register.html
│   ├── script.js
│   ├── styles.css
│   └── theme.js
│
├── requirements.txt
├── test_model.py
├── train_model.py
├── .gitignore
└── README.md
```

---

# 🔄 How XRayAI Works

### 1️⃣ User Authentication

Users can register and log into the platform.

```text
Register
   ↓
Login
   ↓
Authenticated Dashboard
```

### 2️⃣ Upload X-Ray

The user uploads a chest X-ray through the web interface.

### 3️⃣ AI Analysis

The backend processes the image and sends it through the trained DenseNet121 model.

```text
X-Ray
  ↓
Preprocessing
  ↓
DenseNet121
  ↓
Classification
  ↓
Prediction
```

### 4️⃣ Result Generation

The system produces a classification result:

```text
COVID
NORMAL
PNEUMONIA
```

### 5️⃣ Store Analysis

The analysis can be associated with the user's account and stored in the database.

### 6️⃣ Generate Report

The system can generate a structured PDF report containing the analysis information.

### 7️⃣ Review History

Users can access previous analyses through the history interface.

### 8️⃣ AI Assistant

The platform also provides an integrated AI assistant to help users interact with the system and understand information related to their analysis.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/RohitKatigar-963/XRayAI.git
cd XRayAI
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

Navigate to the backend directory:

```bash
cd backend
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

The backend will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation can be accessed through the `/docs` endpoint when the development server is running.

---

# 🌐 Running the Frontend

From the `frontend` directory:

```bash
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

---

# 🧪 Model Evaluation

The project includes a dedicated evaluation script:

```bash
python test_model.py
```

The evaluation pipeline loads the trained model and evaluates it against the test dataset.

---

# 🏋️ Model Training

Training can be performed using:

```bash
python train_model.py
```

The training pipeline supports model training and checkpoint-based progress recovery.

This is particularly useful when training on CPU hardware because long-running training sessions can be interrupted without necessarily losing all previous progress.

---

# 🔐 Security & Privacy Considerations

Because XRayAI deals with medical imaging, security and privacy are important design considerations.

The project includes application-level authentication and separates user-related data from the frontend interface.

For real-world deployment, additional measures would be required, including:

* HTTPS
* Secure authentication tokens
* Password hashing
* Proper authorization
* Secure file storage
* Encryption
* Audit logging
* Medical-data privacy compliance
* Input validation
* Rate limiting
* Secure cloud infrastructure

---

# ⚠️ Medical Disclaimer

**XRayAI is an educational and research project and is NOT a medical diagnostic device.**

The predictions generated by the model should not be used as a substitute for professional medical evaluation.

The reported performance represents evaluation on the project's dataset/test split and does not establish clinical effectiveness, generalization to all patient populations, or regulatory approval.

Any real-world clinical deployment would require extensive external validation, clinical testing, appropriate regulatory review, and evaluation by qualified medical professionals.

---

# 🎯 Project Goals

XRayAI was built with the following goals:

* Explore practical applications of AI in medical imaging
* Build an end-to-end AI product rather than only a standalone model
* Apply transfer learning to medical image classification
* Connect deep learning inference with a production-style backend
* Build a usable web interface around an AI model
* Implement authentication and user-specific history
* Generate structured AI-assisted reports
* Explore AI assistants alongside medical imaging workflows

---

# 🚀 Future Improvements

Potential future improvements include:

* 📍 Explainable AI using Grad-CAM
* 🫁 Lung-region segmentation
* 📊 Confidence visualization
* 📈 Model performance dashboard
* 🔬 Multi-disease detection
* 🧠 More advanced medical foundation models
* ☁️ Cloud deployment
* ⚡ GPU inference
* 🔐 Production-grade authentication
* 🗄️ Scalable cloud database
* 📦 Containerized deployment with Docker
* 📡 Monitoring and observability
* 🧪 External validation datasets
* 👨‍⚕️ Clinician-oriented workflow
* 📱 Mobile-friendly experience

---

# 🧠 What Makes XRayAI Different?

Many machine-learning projects stop at:

```text
Dataset → Train Model → Accuracy
```

XRayAI attempts to build the **complete AI application layer**:

```text
                  ┌───────────────┐
                  │   User        │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Web Interface │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ FastAPI API   │
                  └───────┬───────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          ┌─────────────┐   ┌─────────────┐
          │ AI Model    │   │  Database   │
          └──────┬──────┘   └──────┬──────┘
                 │                 │
                 └────────┬────────┘
                          ▼
                  ┌───────────────┐
                  │ AI Result     │
                  │ + PDF Report  │
                  └───────────────┘
```

The focus is therefore not only on model accuracy, but on demonstrating how an AI model can be integrated into a **complete software product**.

---

# 📌 Current Status

| Component                | Status                |
| ------------------------ | --------------------- |
| Medical DenseNet121      | ✅ Implemented         |
| COVID Classification     | ✅ Implemented         |
| Normal Classification    | ✅ Implemented         |
| Pneumonia Classification | ✅ Implemented         |
| Model Training           | ✅ Completed           |
| Model Evaluation         | ✅ Completed           |
| 91.98% Test Accuracy     | ✅ Achieved            |
| FastAPI Backend          | ✅ Implemented         |
| Frontend                 | ✅ Implemented         |
| Authentication           | ✅ Implemented         |
| Analysis History         | ✅ Implemented         |
| PDF Reports              | ✅ Implemented         |
| AI Assistant             | ✅ Integrated          |
| GitHub Repository        | 🚀 Active Development |

---

# 👨‍💻 Author

**Rohit Katigar**

Computer Science & Engineering

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Generative AI
* 🧩 AI Agents
* 🩻 AI in Healthcare
* 🏗️ AI System Architecture
* ☁️ Cloud & Deployment
* 🔬 Applied Machine Learning

---

# ⭐ If You Find This Project Interesting

Consider giving the repository a ⭐ on GitHub!

Feedback, suggestions, and contributions are welcome.

---

## 📜 License

This project is intended primarily for educational and research purposes. Add an appropriate open-source license to the repository if you intend to permit redistribution or modification.
