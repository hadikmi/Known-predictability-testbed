"""Command-line entry point:  knownpred <command> [options]

Commands
--------
part-a      run the Part A experiments            (knownpred.experiments.part_a)
download    fetch / extract the Part B data       (knownpred.data.download)
part-b      preprocess and evaluate Part B        (knownpred.experiments.part_b)
figures     make figures and tables from runs/    (knownpred.plotting)
data-docs   regenerate docs/data.md from the data-source registry
"""

from __future__ import annotations

import sys


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "part-a":
        from .experiments.part_a import main as run
    elif cmd == "part-b":
        from .experiments.part_b import main as run
    elif cmd == "download":
        from .data.download import main as run
    elif cmd == "figures":
        from .plotting import main as run
    elif cmd == "data-docs":
        from pathlib import Path

        from .data.sources import data_markdown

        path = Path("docs/data.md")
        head = (
            path.read_text().split("| Dataset |")[0]
            if path.exists()
            else "# Data sources (Part B)\n\n"
        )
        path.write_text(head + data_markdown()[data_markdown().index("| Dataset |") :])
        print(f"wrote {path}")
        return
    else:
        raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")
    run(rest)


if __name__ == "__main__":
    main()
