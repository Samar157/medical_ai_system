from django.shortcuts import render
from django.conf import settings
from predictor.nlp_parser import extract_features
from predictor.ocr_utils import image_to_text, pdf_to_text

import joblib
import os

MODEL_PATH = os.path.join(settings.BASE_DIR, "predictor", "heart_model.pkl")
model = joblib.load(MODEL_PATH)

def home(request):
    text = ""

    if request.method == "POST":

        # Text input
        if request.POST.get("report_text"):
            text = request.POST["report_text"]

        # File upload
        if request.FILES.get("file"):
            f = request.FILES["file"]
            path = os.path.join(settings.BASE_DIR, "predictor", "temp_" + f.name)

            with open(path, "wb+") as dest:
                for chunk in f.chunks():
                    dest.write(chunk)

            if path.endswith(".pdf"):
                text = pdf_to_text(path)
            else:
                text = image_to_text(path)

        # Extract features
        feats = extract_features(text)

        # FINAL FIX: only real features
        ordered = [
            feats.get("age", 0),
            feats.get("sex", 0),
            feats.get("trestbps", 0),
            feats.get("chol", 0),
            feats.get("fbs", 0),
            feats.get("thalach", 0),
            feats.get("exang", 0),
            feats.get("oldpeak", 0)
        ]

        # Debug (VERY IMPORTANT)
        print("TEXT:", text)
        print("FEATURES:", feats)
        print("MODEL INPUT:", ordered)

        pred = model.predict([ordered])[0]
        prob = model.predict_proba([ordered])[0][1]

        return render(request, "index.html", {
            "result": "High Risk" if pred else "Low Risk",
            "prob": round(prob * 100, 2),
            "text": text
        })

    return render(request, "index.html")