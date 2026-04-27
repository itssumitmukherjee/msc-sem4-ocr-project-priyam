"""
app.py  —  OCR Sudoku Solver (Telegraph Edition)
=================================================
LOCATION : <project_root>/app.py

LOCAL DEV:
    cd <project_root>
    python app.py
    → http://127.0.0.1:5000

RENDER DEPLOYMENT:
    Build : pip install -r requirements.txt
    Start : gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120

ENDPOINTS
---------
  GET  /               → frontend/index.html
  POST /api/ocr        → image upload  → 81-cell board JSON
  POST /api/solve      → board JSON    → solution JSON
  POST /api/validate   → board JSON    → conflict list (live check)
  GET  /api/health     → model status
"""

import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODEL_PATH   = os.path.join(BACKEND_DIR, "model_weights", "digit_cnn.pth")

for p in (ROOT_DIR, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports ────────────────────────────────────────────────────────────────────
from backend.ocr    import extract_sudoku_from_image, extract_sudoku_full, load_model
from backend.solver import solve_sudoku, validate_board

from flask      import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── Startup ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  OCR Sudoku Solver — Telegraph Edition")
print("=" * 60)
print(f"  Root      : {ROOT_DIR}")
print(f"  Frontend  : {FRONTEND_DIR}")
print(f"  Model     : {MODEL_PATH}")
idx_ok = os.path.isfile(os.path.join(FRONTEND_DIR, "index.html"))
print(f"  index.html: {'FOUND ✓' if idx_ok else 'MISSING ✗ — check frontend/'}")
print("=" * 60 + "\n")

# ── Load model ─────────────────────────────────────────────────────────────────
model       = None
MODEL_READY = False

if os.path.isfile(MODEL_PATH):
    try:
        model       = load_model(MODEL_PATH)
        MODEL_READY = True
        print("  OCR mode : PyTorch CNN  ✓\n")
    except Exception as e:
        print(f"  WARNING  : Model load failed — {e}")
        print("  OCR mode : kNN fallback\n")
else:
    print("  WARNING  : digit_cnn.pth not found.")
    print("  Run  python backend/train.py  to train the model first.")
    print("  OCR mode : kNN fallback  (low accuracy)\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image received."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file received."}), 400

    allowed = {"image/jpeg", "image/jpg", "image/png",
               "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        return jsonify({"error": f"Unsupported file type: {file.content_type}"}), 400

    try:
        img_bytes = file.read()
        result    = extract_sudoku_full(
            img_bytes,
            model=model if MODEL_READY else None,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception:
        app.logger.exception("OCR error")
        return jsonify({"error": "OCR failed. Try a clearer image."}), 500


@app.route("/api/solve", methods=["POST"])
def solve_endpoint():
    data = request.get_json(silent=True)
    if not data or "board" not in data:
        return jsonify({"error": "Send JSON with key 'board'."}), 400
    board = data["board"]
    if not isinstance(board, list) or len(board) != 81:
        return jsonify({"error": "'board' must be a list of 81 integers."}), 400
    if not all(isinstance(v, int) and 0 <= v <= 9 for v in board):
        return jsonify({"error": "All values must be integers 0–9."}), 400
    return jsonify(solve_sudoku(board))


@app.route("/api/validate", methods=["POST"])
def validate_endpoint():
    """Called on every cell edit — returns conflicts without solving."""
    data = request.get_json(silent=True)
    if not data or "board" not in data:
        return jsonify({"error": "Send JSON with key 'board'."}), 400
    board = data["board"]
    if not isinstance(board, list) or len(board) != 81:
        return jsonify({"error": "'board' must be a list of 81 integers."}), 400
    return jsonify(validate_board(board))


@app.route("/api/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY,
        "ocr_mode":    "cnn" if MODEL_READY else "knn_fallback",
        "index_found": os.path.isfile(os.path.join(FRONTEND_DIR, "index.html")),
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"  Open: http://127.0.0.1:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)