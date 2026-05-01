Emotion Detection API (FastAPI)

Quick start

1. Create virtual env and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

2. Put your V1 model file in `model/` (e.g. `model/emotion_detection_v1.keras`) or set `MODEL_PATH` env var.
3. Set API key (env var `API_KEY`) or edit `.env`

Run locally (development):

```bash
$env:API_KEY='your_key_here'
$env:MODEL_PATH='model/emotion_detection_v1.keras'
uvicorn model_api.main:app --host 0.0.0.0 --port 8000 --reload
```

Request example (curl):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-KEY: your_key_here" \
  -F "file=@/path/to/image.jpg"
```

Reload model after replacing file:

```bash
curl -X POST "http://localhost:8000/reload" -H "X-API-KEY: your_key_here"
```

Replacing model:
- Replace the file pointed by `MODEL_PATH` (e.g. overwrite `model/emotion_detection_v1.keras`) and call `/reload` endpoint.
- Or set `MODEL_PATH` to point to a different file and restart the server.

Notes:
- `main.py` includes minimal re-implementations of `LabelSmoothingLoss` and `SqueezeExcitation` to support loading the V1 model.
- If your model uses additional custom objects, add them to `custom_objects` in `main.py`.
