#!/usr/bin/env python3
"""Lightweight demo server — serves experiment reports on 0.0.0.0:8080.

Usage:
    python serve_demos.py [--port 8080] [--host 0.0.0.0]
"""

import argparse
import http.server
import functools
import os

DEMOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos")


def main():
    parser = argparse.ArgumentParser(description="SNKS demo server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DEMOS_DIR)
    server = http.server.HTTPServer((args.host, args.port), handler)
    print(f"Serving demos at http://{args.host}:{args.port}/")
    print(f"  Directory: {DEMOS_DIR}")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
