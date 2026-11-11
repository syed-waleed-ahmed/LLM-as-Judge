"""The release build points the README's relative links at GitHub for PyPI."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
REPO, REF = "owner/name", "v9.9.9"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pypi_readme", ROOT / "scripts" / "pypi_readme.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pypi_readme = _load_script()


def test_links_images_and_fragments():
    text = (
        "See [the roadmap](ROADMAP.md) and [Supabase](DEPLOYMENT.md#supabase).\n"
        "![Shot](docs/assets/shot.png) [![Wrapped](./img/a.PNG)](docs/big.png)\n"
    )
    out = pypi_readme.absolutize(text, REPO, REF)
    assert "[the roadmap](https://github.com/owner/name/blob/v9.9.9/ROADMAP.md)" in out
    assert "(https://github.com/owner/name/blob/v9.9.9/DEPLOYMENT.md#supabase)" in out
    raw = "https://raw.githubusercontent.com/owner/name/v9.9.9"
    assert f"![Shot]({raw}/docs/assets/shot.png)" in out
    # A link wrapping an image: both targets are rewritten.
    assert f"[![Wrapped]({raw}/img/a.PNG)]({raw}/docs/big.png)" in out


def test_absolute_links_anchors_and_mail_are_untouched():
    text = "[a](https://x.org/p) [b](#usage) [c](mailto:a@b.c) [d](//cdn.x/y.png)"
    assert pypi_readme.absolutize(text, REPO, REF) == text


def test_the_real_readme_has_no_relative_targets_left(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text((ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert pypi_readme.main(["--repo", REPO, "--ref", REF, str(readme)]) == 0
    targets = re.findall(r"\]\(([^)\s]+)\)", readme.read_text(encoding="utf-8"))
    relative = [t for t in targets if not re.match(r"^(?:https?:|#|mailto:)", t)]
    assert targets and not relative
