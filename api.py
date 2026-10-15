"""Optional Flask API for model inference. Train a model from the Streamlit Model Center first."""
from flask import Flask, request, jsonify
from src.model import load_model, predict_rows

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok", "service":"Sentinel Predictive Maintenance API"})

@app.post("/predict")
def predict():
    bundle = load_model()
    if bundle is None:
        return jsonify({"error":"Model not found. Train the model in the dashboard first."}), 503
    payload = request.get_json(silent=True)
    required = ["temperature","vibration","current","pressure","load","age_days"]
    if not isinstance(payload, dict) or any(k not in payload for k in required):
        return jsonify({"error":"JSON body must include: " + ", ".join(required)}), 400
    try:
        row = {k: float(payload[k]) for k in required}
        pred, prob = predict_rows(bundle, [row])
        return jsonify({
            "asset_status":"failure_risk" if int(pred[0]) else "monitor",
            "predicted_failure":int(pred[0]),
            "failure_probability":round(float(prob[0]), 4),
            "note":"Demo prediction based on synthetic training labels; not for safety-critical use."
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error":f"Invalid numeric sensor value: {exc}"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
