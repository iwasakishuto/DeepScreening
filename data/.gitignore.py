# coding: utf-8
import os
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    with open(os.path.join(here, ".gitignore"), mode="w") as f:
        p = Path(here)
        for path in p.glob("**/*"):
            if path.stat().st_size > 100000000:
                f.write(str(path.relative_to(p.cwd())) + "\n")
