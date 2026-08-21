"""Top-level entry point: run the full perturbation-rejection pipeline and build the report.

Usage: python run.py   (run from the perturbation_rejection/ directory)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pipeline
import report
import json

if __name__ == "__main__":
    pipeline.main()
    results = json.load(open("../outputs/results.json"))
    heavy = json.load(open("../outputs/heavy.json"))
    html = report.build_report(results, heavy)
    with open("../outputs/report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote outputs/report.html", len(html), "bytes")
