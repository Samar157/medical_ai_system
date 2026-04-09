import re

def extract_features(text):
    text = text.lower()

    def find(pattern, group_index=1, default=0):
        match = re.search(pattern, text)
        return float(match.group(group_index)) if match else default

    features = {
        "age": find(r'age[: ]+(\d+)', 1),

        "sex": 1 if "male" in text else 0,

        # FIXED HERE → use group 2
        "trestbps": find(r'(bp|blood pressure)[: ]+(\d+)', 2),

        "chol": find(r'(chol|cholesterol)[: ]+(\d+)', 2),

        "fbs": 1 if "diabetes" in text or "sugar" in text else 0,

        "thalach": find(r'(heart rate|hr)[: ]+(\d+)', 2),

        "exang": 1 if "chest pain" in text else 0,

        "oldpeak": find(r'oldpeak[: ]+(\d+\.?\d*)', 1)
    }

    return features