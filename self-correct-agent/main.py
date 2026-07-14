"""Project CLI entry point."""

from __future__ import annotations

import sys

from eval.run_eval import main


if __name__ == "__main__":
    # Preserve the project-level command in the original project brief.
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        del sys.argv[1]
    main()
