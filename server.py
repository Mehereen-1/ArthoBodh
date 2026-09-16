"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) Launcher
Root entry point to run the ArthoBodh WSD API and interactive Web UI.
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"\n=======================================================")
    print(f"   ArthoBodh Interface running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
