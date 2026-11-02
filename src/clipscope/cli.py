from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .encoder import MODELS, OpenCLIPEncoder
from .evaluation import evaluate, write_report
from .index import VectorIndex, build_index
from .metrics import paired_comparison
from .synthetic import generate


def main():
    parser = argparse.ArgumentParser(description="CLIPScope retrieval laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("demo-data", help="Generate original 48-image shape diagnostic")
    synth.add_argument("--out", type=Path, default=Path("data/shapes"))
    for name in ("index", "evaluate", "suite"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", choices=MODELS, default="b32-openai")
        sub.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        sub.add_argument("--cache", default=".cache/models")
        sub.add_argument("--index", type=Path, default=Path("data/index.sqlite"))
        if name in ("index", "suite"):
            sub.add_argument("--manifest", type=Path, default=Path("data/shapes/manifest.jsonl"))
            sub.add_argument("--batch-size", type=int, default=8)
        if name in ("evaluate", "suite"):
            sub.add_argument("--queries", type=Path, default=Path("data/shapes/queries.jsonl"))
            sub.add_argument("--split", choices=("dev", "test"), default="test")
            sub.add_argument("--out", type=Path, default=Path("experiments/results" if name == "suite" else "experiments/results/run.json"))
        if name == "evaluate":
            sub.add_argument("--ensemble", action="store_true")
            sub.add_argument("--direction", choices=("text-to-image", "image-to-text"), default="text-to-image")
    compare = commands.add_parser("compare")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "demo-data":
        generate(args.out)
        print(f"Generated diagnostic in {args.out}")
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("clipscope.api:create_app", factory=True, host=args.host, port=args.port, workers=1)
    elif args.command == "compare":
        print(json.dumps(paired_comparison(json.loads(args.a.read_text()), json.loads(args.b.read_text())), indent=2))
    else:
        encoder = OpenCLIPEncoder(args.model, args.device, args.cache)
        if args.command in ("index", "suite"):
            print(json.dumps(build_index(args.manifest, args.index, encoder, args.batch_size), indent=2))
        if args.command in ("evaluate", "suite"):
            index = VectorIndex(args.index, encoder.fingerprint)
            if args.command == "evaluate":
                report = evaluate(index, encoder, args.queries, args.split, args.ensemble, args.direction)
                write_report(report, args.out)
                print(f"Saved {args.out}")
            else:
                reports = []
                for mode in (False, True):
                    report = evaluate(index, encoder, args.queries, args.split, mode)
                    target = args.out / f"{args.model}-{args.split}-{'ensemble' if mode else 'baseline'}.json"
                    write_report(report, target)
                    reports.append(report)
                    print(f"Saved {target}")
                report = evaluate(index, encoder, args.queries, args.split, direction="image-to-text")
                write_report(report, args.out / f"{args.model}-{args.split}-image-to-text.json")
                (args.out / f"{args.model}-{args.split}-comparison.json").write_text(
                    json.dumps(paired_comparison(*reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
