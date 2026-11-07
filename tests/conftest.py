import os
import sys
import tempfile
from pathlib import Path

# Keep test predictions out of the real usage log
os.environ["FLYRELY_USAGE_LOG"] = str(Path(tempfile.mkdtemp()) / "usage_log.jsonl")
os.environ["FLYRELY_PREDICTION_DB"] = str(Path(tempfile.mkdtemp()) / "predictions.sqlite")
os.environ.pop("DATABASE_URL", None)
os.environ.pop("WEATHER_API_KEY", None)
os.environ["WEATHER_API_PROVIDER"] = "none"  # no network in tests; weather tests inject fixtures

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
