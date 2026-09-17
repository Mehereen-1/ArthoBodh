"""
ArthoBodh launcher: python server.py  ->  http://127.0.0.1:8000
"""

import os

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nArthoBodh running at http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
