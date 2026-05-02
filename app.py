"""
app.py  —  OCR Sudoku Solver (Telegraph Edition)
=================================================
HOW TO RUN:
    cd <project_root>
    python app.py
    Open: http://127.0.0.1:5000

ENDPOINTS
─────────
  GET  /             → frontend/index.html
  POST /api/ocr      → image upload → 81-cell board JSON + pipeline images
  POST /api/solve    → board JSON   → solution JSON
  POST /api/validate → board JSON   → conflict list (live check)
  GET  /api/health   → model + server status
"""

import os
import sys
import logging
import traceback

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODEL_PATH   = os.path.join(BACKEND_DIR, "model_weights", "digit_cnn.pth")

for p in (ROOT_DIR, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("sudoku_ocr")

# ── Flask ──────────────────────────────────────────────────────────────────────
from flask      import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Project imports ────────────────────────────────────────────────────────────
from backend.ocr    import extract_sudoku_full, extract_sudoku_from_image, load_model
from backend.solver import solve_sudoku, validate_board

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Max upload: 10 MB (newspaper photos are typically 2-5 MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Allowed MIME types — covers all common browser variations
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/pjpeg",   # JPEG (various browser labels)
    "image/png",  "image/x-png",                 # PNG
    "image/webp",                                 # WEBP
    "image/bmp",  "image/x-bmp",                 # BMP
    "application/octet-stream",                   # Some browsers send this for any file
    "",                                           # No content-type header at all
}

# Allowed file extensions — used as fallback when MIME type is missing/wrong
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP — print status and load model
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  OCR Sudoku Solver — Telegraph Edition")
print("=" * 60)
print(f"  Root     : {ROOT_DIR}")
print(f"  Frontend : {FRONTEND_DIR}")
print(f"  Model    : {MODEL_PATH}")
idx_ok = os.path.isfile(os.path.join(FRONTEND_DIR, "index.html"))
print(f"  index.html: {'FOUND ✓' if idx_ok else 'MISSING ✗ — check frontend/'}")
print("=" * 60 + "\n")

model       = None
MODEL_READY = False

if os.path.isfile(MODEL_PATH):
    try:
        model       = load_model(MODEL_PATH)
        MODEL_READY = True
        print("  OCR mode : DigitCNN (PyTorch)  ✓\n")
    except RuntimeError as e:
        print(f"  ERROR : Model weights do not match the current architecture.")
        print(f"  Fix   : Delete digit_cnn.pth and retrain:  python backend/train.py")
        print(f"  OCR mode : kNN fallback\n")
    except Exception as e:
        print(f"  WARNING : Model load failed — {e}")
        print(f"  OCR mode : kNN fallback\n")
else:
    print("  WARNING : digit_cnn.pth not found.")
    print("  Run     : python backend/train.py  to train the model first.")
    print("  OCR mode : kNN fallback  (lower accuracy)\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _error(message: str, code: int = 400):
    """Return a consistent JSON error response."""
    return jsonify({"error": message}), code


def _validate_board_input(data) -> tuple:
    """
    Validate the JSON 'board' field.
    Returns (board_list, None) on success or (None, error_response) on failure.
    """
    if not data or "board" not in data:
        return None, _error("Send JSON with key 'board'.")
    board = data["board"]
    if not isinstance(board, list) or len(board) != 81:
        return None, _error("'board' must be a list of exactly 81 integers.")
    if not all(isinstance(v, int) and 0 <= v <= 9 for v in board):
        return None, _error("All board values must be integers 0–9.")
    return board, None


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(413)
def too_large(e):
    return _error("Image too large. Maximum allowed size is 10 MB.", 413)

@app.errorhandler(404)
def not_found(e):
    return _error("Endpoint not found.", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return _error("Method not allowed.", 405)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    """
    Upload a sudoku image.
    Returns 81-cell board + pipeline visualisation images.
    """
    if "image" not in request.files:
        return _error("No image received. Send file with field name 'image'.")

    file = request.files["image"]
    if not file or not file.filename:
        return _error("Empty file received.")

    # Determine file type — check MIME type first, then fall back to extension.
    # Browsers are inconsistent: Chrome may send 'image/jpeg', Firefox 'image/jpg',
    # mobile browsers sometimes send 'application/octet-stream' or nothing at all.
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    filename_ext = os.path.splitext(file.filename or "")[1].lower()

    type_ok = content_type in ALLOWED_TYPES
    ext_ok  = filename_ext in ALLOWED_EXTS

    if not type_ok and not ext_ok:
        return _error(
            f"Unsupported file. Please upload a JPEG, PNG, WEBP or BMP image. "
            f"Got type='{content_type}', ext='{filename_ext}'.", 415)

    try:
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return _error("Received an empty file.")

        result = extract_sudoku_full(
            img_bytes,
            model=model if MODEL_READY else None,
        )

        log.info(f"OCR complete — {result.get('digit_count', 0)} digits  "
                 f"mode={result.get('mode', '?')}")
        return jsonify(result)

    except ValueError as e:
        log.warning(f"OCR bad image: {e}")
        return _error(str(e), 422)

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("OCR pipeline error:\n" + tb)
        # Return detailed error in debug mode so dev can diagnose quickly
        import os as _os
        if _os.environ.get("FLASK_DEBUG", "1") == "1":
            return _error(f"OCR failed: {type(exc).__name__}: {exc}", 500)
        return _error(
            "OCR processing failed. Try a clearer, well-lit photo of the grid.",
            500)


@app.route("/api/solve", methods=["POST"])
def solve_endpoint():
    """Solve a sudoku board."""
    board, err = _validate_board_input(request.get_json(silent=True))
    if err:
        return err
    try:
        result = solve_sudoku(board)
        log.info(f"Solve: solved={result.get('solved')}")
        return jsonify(result)
    except Exception:
        log.error("Solve error:\n" + traceback.format_exc())
        return _error("Solver encountered an unexpected error.", 500)


@app.route("/api/validate", methods=["POST"])
def validate_endpoint():
    """Live conflict check — called on every cell edit."""
    board, err = _validate_board_input(request.get_json(silent=True))
    if err:
        return err
    try:
        return jsonify(validate_board(board))
    except Exception:
        log.error("Validate error:\n" + traceback.format_exc())
        return _error("Validation encountered an unexpected error.", 500)


@app.route("/api/health", methods=["GET"])
def health():
    """Model and server status."""
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY,
        "ocr_mode":    "cnn" if MODEL_READY else "knn_fallback",
        "index_found": os.path.isfile(os.path.join(FRONTEND_DIR, "index.html")),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"  Open: http://127.0.0.1:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)