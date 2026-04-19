from flask import Flask, request, jsonify, render_template
import tempfile
import os

from prediction import predict_from_csv

app = Flask(__name__)


# Route to load the frontend
@app.route("/")
def home():
    return render_template("index.html")


# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as tmp:

            file.save(tmp.name)
            path = tmp.name

        print("File received:", path)

        # Run prediction
        data, suspicious = predict_from_csv(path)

        print("Prediction completed")

        response = {
    "results": (
        suspicious
        .replace([float("inf"), float("-inf")], 0)
        .fillna("")
        .to_dict(orient="records")
    ),
    "total_records": int(len(data)),
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

        return jsonify(response)

    except Exception as e:

        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if 'path' in locals() and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )