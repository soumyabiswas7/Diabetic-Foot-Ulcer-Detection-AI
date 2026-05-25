if prediction[0][0] >= 0.5:
        result = "Ulcer"
        confidence = prediction[0][0]