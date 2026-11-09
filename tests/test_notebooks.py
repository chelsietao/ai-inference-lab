import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))
REPO = "chelsietao/ai-inference-lab"


def test_three_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == [
        "01_pytorch_inference.ipynb",
        "02_cpu_vs_gpu.ipynb",
        "03_batch_size_benchmark.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_structure(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert any(c["cell_type"] == "markdown" for c in cells)
    code = [c for c in cells if c["cell_type"] == "code"]
    assert code
    # No committed outputs: results must come from a real run.
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code)
    for c in code:  # syntactically valid python (ignoring magics)
        src = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
        compile(src, path.name, "exec")
    text = "".join("".join(c["source"]) for c in cells)
    assert "ai_inference_lab" in text and "git+https://github.com/" + REPO in text
    assert "/Users/" not in text


def test_readme_has_colab_link_for_each_notebook():
    readme = (NB_DIR.parent / "README.md").read_text(encoding="utf-8")
    for p in NOTEBOOKS:
        assert f"colab.research.google.com/github/{REPO}/blob/main/notebooks/{p.name}" in readme
