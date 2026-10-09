"""SafeCDS V2: Local Presentation Web Server Runner.

Launches the local processing backend and interactive presentation UI at http://127.0.0.1:8000.
"""

import argparse
import os
import sys
import webbrowser

try:
    import uvicorn
except ImportError:
    print("[ERROR] uvicorn is not installed. Run: pip install uvicorn fastapi")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SafeCDS V2 Presentation Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")

    args = parser.parse_args()

    print("=" * 75)
    print("      SafeCDS V2: VERIFIABLE NEURO-SYMBOLIC CLINICAL REASONING     ")
    print("=" * 75)
    print(f"[*] Starting Local Web Presentation Server...")
    print(f"[*] Local Web Access:    http://{args.host}:{args.port}")
    print(f"[*] API Documentation:   http://{args.host}:{args.port}/docs")
    print(f"[*] Engine Environment:  On-Premise (Local Ollama + Isolated HermiT DL)")
    print("=" * 75)
    print("Press Ctrl+C to terminate the local server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(f"http://{args.host}:{args.port}")
        except Exception:
            pass

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
