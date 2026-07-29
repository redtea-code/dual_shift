"""Compact NDJSON debug logger for agent debug sessions."""
from __future__ import annotations

import json
import os
import time

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug-49032c.log')
_SESSION_ID = '49032c'


def agent_log(
        location: str,
        message: str,
        data: dict | None = None,
        *,
        hypothesis_id: str = '',
        run_id: str = 'pre-fix',
):
    # #region agent log
    payload = {
        'sessionId': _SESSION_ID,
        'runId': run_id,
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data or {},
        'timestamp': int(time.time() * 1000),
    }
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, default=str) + '\n')
    except OSError:
        pass
    # #endregion
