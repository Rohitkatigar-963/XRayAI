from fastapi.security import OAuth2PasswordRequestForm
from database import SessionLocal, engine
from models import User, Prediction
from database import Base
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# 🔽 ADD THESE BELOW
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from pdf_generator import create_pdf_report
from fastapi.responses import StreamingResponse
from io import BytesIO
# 🔼 END ADD

import torch
import torch.nn as nn
import torchxrayvision as xrv

from PIL import Image

import io
import os
import cv2
import numpy as np

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Chest X-ray API"
)

# Stores the latest prediction for report download
latest_prediction = None


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

model = None


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# LOAD MEDICAL PRETRAINED DENSENET121
# ============================================================

def load_xray_model():

    global model

    # --------------------------------------------------------
    # Allow backend to start while model is still training
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("=" * 60)
        print("WARNING: Trained X-ray model not found.")
        print(f"Expected model path: {MODEL_PATH}")
        print("Backend will start without prediction support.")
        print("=" * 60)

        model = None

        return


    print("=" * 60)
    print("Loading trained Medical DenseNet121...")
    print("=" * 60)


    # --------------------------------------------------------
    # Create same TorchXRayVision architecture used in training
    # --------------------------------------------------------

    xray_model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )


    # Disable original 18-class operating thresholds
    xray_model.op_threshs = None


    # --------------------------------------------------------
    # Create EXACT same classifier used during training
    # --------------------------------------------------------

    in_features = (
        xray_model.classifier.in_features
    )

    xray_model.classifier = nn.Sequential(

        nn.Linear(
            in_features,
            512
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            0.4
        ),

        nn.Linear(
            512,
            len(CLASS_NAMES)
        )

    )


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    xray_model.load_state_dict(
        state_dict
    )


    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    xray_model = xray_model.to(
        DEVICE
    )

    xray_model.eval()


    model = xray_model


    print(
        "Medical DenseNet121 loaded successfully."
    )

    print(
        f"Device: {DEVICE}"
    )

    print("=" * 60)


# Load model when backend starts
load_xray_model()


# ============================================================
# MEDICAL X-RAY PREPROCESSING
# ============================================================

def preprocess_xray(image):

    # Convert PIL image to grayscale NumPy array
    image = image.convert("L")

    image = np.array(
        image
    ).astype(
        np.float32
    )

    # Ensure contiguous array
    image = np.ascontiguousarray(
        image
    )

    # Same TorchXRayVision normalization as training
    image = xrv.datasets.normalize(
        image,
        255
    )

    # Add channel dimension
    # H,W -> 1,H,W
    image = image[
        None,
        :,
        :
    ]

    # Same center crop used during training
    image = xrv.datasets.XRayCenterCrop()(
        image
    )

    # Same resize used during training
    image = xrv.datasets.XRayResizer(
        224
    )(
        image
    )

    # Convert to PyTorch tensor
    image = torch.from_numpy(
        image
    ).float()

    # Add batch dimension
    # 1,224,224 -> 1,1,224,224
    image = image.unsqueeze(
        0
    )

    return image.to(
        DEVICE
    )
# ================= AUTH FUNCTIONS =================
# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

SECRET_KEY = "supersecretkey"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="login"
)


# ============================================================
# DATABASE SESSION
# ============================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()
import hashlib

def hash_password(password: str):
    # First hash with SHA256
    sha256_password = hashlib.sha256(password.encode()).hexdigest()
    # Then bcrypt
    return pwd_context.hash(sha256_password)

def verify_password(plain_password, hashed_password):
    sha256_password = hashlib.sha256(plain_password.encode()).hexdigest()
    return pwd_context.verify(sha256_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user

def predict_image(image: Image.Image):

    # Check if trained model is available
    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "X-ray AI model is currently unavailable. "
                "The new medical model is still being trained."
            )
        )


    # Preprocess exactly like training
    img = preprocess_xray(
        image
    )


    # Run inference
    with torch.no_grad():

        outputs = model(
            img
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]


    # Convert probabilities to NumPy
    probs = probabilities.cpu().numpy()


    # Find highest probability
    pred_idx = int(
        np.argmax(
            probs
        )
    )


    # Create probability dictionary
    probability_dict = {

        CLASS_NAMES[i]:
            float(
                probs[i] * 100
            )

        for i in range(
            len(CLASS_NAMES)
        )

    }


    # Prediction
    prediction = CLASS_NAMES[
        pred_idx
    ]


    # Confidence
    confidence = float(
        probs[pred_idx] * 100
    )


    # Risk level
    if confidence >= 85:

        risk = "High"

    elif confidence >= 60:

        risk = "Medium"

    else:

        risk = "Low"


    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,

        "risk":
            risk

    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(image)
    global latest_prediction
    latest_prediction = result

    # Save prediction to database
    new_prediction = Prediction(
    image_name=file.filename,
    prediction=result["prediction"],
    confidence=result["confidence"],

    covid_probability=result["probabilities"].get("COVID", 0),
    pneumonia_probability=result["probabilities"].get("PNEUMONIA", 0),
    normal_probability=result["probabilities"].get("NORMAL", 0),

    user_id=current_user.id
)

    db.add(new_prediction)
    db.commit()

    return result


# ================= REGISTER =================

from pydantic import BaseModel

# ================= SCHEMA =================

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ================= REGISTER =================

@app.post("/register")
def register(user: RegisterRequest,
             db: Session = Depends(get_db)):

    try:
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")

        new_user = User(
            name=user.name,
            email=user.email,
            password=hash_password(user.password)
        )

        db.add(new_user)
        db.commit()

        return {"message": "User created successfully"}

    except Exception as e:
        print("REGISTER ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):

    # Find user by email (username field)
    db_user = db.query(User).filter(User.email == form_data.username).first()

    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Verify password
    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    access_token = create_access_token(
        data={"user_id": db_user.id}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/download-report/{prediction_id}")
def download_report(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    confidence = prediction.confidence

    if confidence >= 85:
        risk = "High"
    elif confidence >= 60:
        risk = "Medium"
    else:
        risk = "Low"

    pdf_buffer = create_pdf_report(
    user_name=current_user.name,
    prediction=prediction.prediction,
    confidence=prediction.confidence,
    risk=risk,
    probabilities={
    "COVID": prediction.covid_probability,
    "PNEUMONIA": prediction.pneumonia_probability,
    "NORMAL": prediction.normal_probability
}
)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            f"attachment; filename=report_{prediction_id}.pdf"
        }
    )

@app.get("/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(
        Prediction.timestamp.desc()
    ).all()

    history = []

    for prediction in predictions:
        history.append({
    "id": prediction.id,
    "image_name": prediction.image_name,
    "prediction": prediction.prediction,
    "confidence": prediction.confidence,

    "covid_probability": prediction.covid_probability,
    "pneumonia_probability": prediction.pneumonia_probability,
    "normal_probability": prediction.normal_probability,

    "timestamp": prediction.timestamp
})

    return history


@app.delete("/history/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    db.delete(prediction)
    db.commit()

    return {
        "message": "Prediction deleted successfully"
    }



@app.get("/download-report")
def download_latest_report(
    current_user: User = Depends(get_current_user)
):
    global latest_prediction

    if latest_prediction is None:
        raise HTTPException(
            status_code=400,
            detail="No prediction available"
        )

    pdf_buffer = create_pdf_report(
    user_name=current_user.name,
    prediction=latest_prediction["prediction"],
    confidence=latest_prediction["confidence"],
    risk=latest_prediction["risk"],
    probabilities=latest_prediction["probabilities"]
)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            "attachment; filename=xray_report.pdf"
        }
    )


@app.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)
):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email
    }