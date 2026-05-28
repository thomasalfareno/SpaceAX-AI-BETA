"""NDJSON debug logger — session 859b31."""
import json
import os
import time

LOG_PATH = "/home/gracecia/Musik/SpaceaxAiDebug/.cursor/debug-859b31.log"
SESSION_ID = "859b31"


def agent_log(
    location: str,
    message: str,
    data: dict | None = None,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": SESSION_ID,
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "hypothesisId": hypothesis_id,
            "runId": run_id,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
