"""
ArthoBodh launcher: python server.py  ->  http://127.0.0.1:8000
"""

import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nArthoBodh running at http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
