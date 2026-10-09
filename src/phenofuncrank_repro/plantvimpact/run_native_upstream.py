"Run crop-specific candidate, mutation-rate and label calculations."
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROUTES = {
    "candidates": "candidate_universe",
    "rates": "mutation_opportunity",
    "coding-labels": "coding_labels",
    "noncoding-labels": "noncoding_labels",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=("rice", "maize", "tomato", "chickpea", "cotton"), required=True)
    parser.add_argument("--step", choices=ROUTES, required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON with the original argument list and optional working_directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    suffix = ROUTES[args.step]
    if args.species == "maize" and args.step == "noncoding-labels":
        suffix = "maize_noncoding_labels"
    if args.species == "cotton":
        if args.step == "candidates":
            suffix = "cotton_candidate_universe"
        elif args.step == "noncoding-labels":
            suffix = "cotton_noncoding_labels"
    module = "phenofuncrank_repro.plantvimpact.native_upstream." + suffix
    if args.species == "cotton" and args.step == "rates":
        module = "phenofuncrank_repro.plantvimpact.labels_loco.cotton_pmut_cli"
    working_directory = Path(config.get("working_directory", "."))
    if not working_directory.is_absolute():
        working_directory = args.config.resolve().parent / working_directory
    arguments = [os.path.expandvars(str(value)) for value in config["arguments"]]
    if args.species == "cotton" and args.step == "rates" and "--calc-script" not in arguments:
        arguments += ["--calc-script", str(Path(__file__).parent / "scripts/labels_loco/cotton_calc_pmut.py")]
    command = [sys.executable, "-B", "-m", module, *arguments]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run(command, cwd=working_directory, env=environment, check=True)


if __name__ == "__main__":
    main()
