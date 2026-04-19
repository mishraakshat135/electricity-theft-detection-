from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import os
import traceback

from prediction import predict_from_csv

app = FastAPI()
from flask import Flask, render_template, request

app = Flask(__name__)

# Home route (loads frontend)
@app.route("/")
def home():
    return render_template("index.html")


# THIS is where the predict route must be
@app.route("/predict", methods=["POST"])
def predict():

    print("Predict route triggered")

    return "Prediction done"


if __name__ == "__main__":
    app.run(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("STEP 1: Request received")

    try:

        contents = await file.read()

        print("STEP 2: File read")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as tmp:

            tmp.write(contents)
            path = tmp.name

        print("STEP 3: File saved at:", path)

        data, suspicious = predict_from_csv(path)
        suspicious["consumer_id"] = suspicious["UserId"]
        results = suspicious.fillna("").to_dict(
    orient="records"
)
        print("STEP 4: Prediction completed")

        response = {
            "results": suspicious.to_dict(
                orient="records"
            ),
            "total_records": len(data),
            "high_risk": int(
                (data["inspection_priority"] == "HIGH").sum()
            ),
            "suspicious_count": int(
                data["inspection_priority"]
                .isin(["HIGH", "MEDIUM"])
                .sum()
            ),
            "estimated_loss": int(
                data["estimated_loss"].sum()
            )
        }

        print("STEP 5: Response ready")
    

        return response

    except Exception as e:

        print("ERROR OCCURRED:")
        traceback.print_exc()

        return {
            "error": str(e)
        }

    finally:

        if 'path' in locals() and os.path.exists(path):
            os.remove(path)