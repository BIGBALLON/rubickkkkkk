"""End-to-end fused-query golden-set evaluation (multipart POST /search).

Slow suite (``@pytest.mark.slow``): loads the MLX model and ingests a
small local fixture tree. The fixture is **not** committed — see
``tests/fixtures/fused_e2e/README.md``.

Golden-set cases use soft top-K expectations (stable across minor drift).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


def _fixture_root() -> Path:
    """Root of the optional fused-e2e corpus (notes/ + images/)."""
    env = os.environ.get("RUBICK_FUSED_FIXTURE_DIR")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "fixtures" / "fused_e2e"
    if not root.is_dir():
        pytest.skip(
            f"fused e2e fixture missing at {root}; "
            "see tests/fixtures/fused_e2e/README.md"
        )
    if not (root / "images" / "cat-portrait.jpg").is_file():
        pytest.skip(
            f"fused e2e fixture incomplete at {root} "
            "(need images/cat-portrait.jpg); see tests/fixtures/fused_e2e/README.md"
        )
    return root


@pytest.fixture(scope="module")
def indexed_app(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Ingest the local fixture into an isolated LanceDB once per module."""
    fixture = _fixture_root()

    data_dir = tmp_path_factory.mktemp("rubick-fused-e2e")
    os.environ["RUBICK_DATA_DIR"] = str(data_dir)

    import importlib
    import sys

    for mod_name in (
        "rubick_backend.settings",
        "rubick_backend.store",
        "rubick_backend.embed",
        "rubick_backend.ingest",
        "rubick_backend.retrieve",
        "rubick_backend.api.search",
        "rubick_backend.main",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from rubick_backend.ingest import ingest_path

    n_files = 0
    for sub in ("notes", "images"):
        subdir = fixture / sub
        if subdir.is_dir():
            stats = ingest_path(subdir)
            n_files += stats["files"]
    assert n_files >= 2, f"expected ≥2 fixture files ingested, got {n_files}"

    from rubick_backend.main import app

    return app


@pytest.fixture()
def client(indexed_app: Any) -> TestClient:
    return TestClient(indexed_app)


def _post_fused(
    client: TestClient,
    *,
    q: str,
    attachment_kind: str | None,
    attachment_path: Path | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    assert attachment_kind is not None and attachment_path is not None
    files = {attachment_kind: (attachment_path.name, attachment_path.read_bytes())}
    data = {"q": q, "limit": str(limit)}
    r = client.post("/search", data=data, files=files)
    assert r.status_code == 200, r.text
    return r.json()["results"]


def _result_paths(results: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for r in results:
        paths = r.get("file_paths", [])
        if paths:
            out.append(Path(paths[0]).name)
    return out


GOLDEN_SET: list[dict[str, Any]] = [
    {
        "name": "image-as-query: cat photo finds itself",
        "q": "",
        "attachment": ("image", "images/cat-portrait.jpg"),
        "k": 3,
        "expected_in_top_k": ["cat-portrait.jpg"],
    },
    {
        "name": "image-as-query: mars hubble finds itself",
        "q": "",
        "attachment": ("image", "images/mars-hubble.jpg"),
        "k": 3,
        "expected_in_top_k": ["mars-hubble.jpg"],
        "forbidden_in_top_k": ["cat-portrait.jpg"],
    },
    {
        "name": "fused T+I: 'space exploration' + mars image keeps mars",
        "q": "space exploration",
        "attachment": ("image", "images/mars-hubble.jpg"),
        "k": 5,
        "expected_in_top_k": ["mars-hubble.jpg"],
        "forbidden_in_top_k": ["cat-portrait.jpg"],
    },
    {
        "name": "fused T+I: 'animal companion' + cat image stays on cat content",
        "q": "animal companion",
        "attachment": ("image", "images/cat-portrait.jpg"),
        "k": 3,
        "expected_in_top_k": ["cat-portrait.jpg"],
        "forbidden_in_top_k": ["jupiter-juno.jpg"],
    },
]


@pytest.mark.parametrize("case", GOLDEN_SET, ids=lambda c: c["name"])
def test_fused_golden_case(client: TestClient, case: dict[str, Any]) -> None:
    fixture = _fixture_root()
    kind, rel = case["attachment"]
    attachment_path = fixture / rel
    if not attachment_path.is_file():
        pytest.skip(f"fixture asset missing: {rel}")

    results = _post_fused(
        client,
        q=case["q"],
        attachment_kind=kind,
        attachment_path=attachment_path,
        limit=max(case.get("k", 10), 10),
    )
    top_k = _result_paths(results)[: case["k"]]

    for expected in case.get("expected_in_top_k", []):
        assert expected in top_k, (
            f"{case['name']!r}: expected {expected!r} in top {case['k']}; got {top_k}"
        )
    for forbidden in case.get("forbidden_in_top_k", []):
        assert forbidden not in top_k, (
            f"{case['name']!r}: did not expect {forbidden!r} in top {case['k']}; got {top_k}"
        )


def test_pure_image_query_matches_ingest_embedding(client: TestClient) -> None:
    """Pure image query should rank the same file at top-1 (~cosine 1.0)."""
    fixture = _fixture_root()
    cat = fixture / "images" / "cat-portrait.jpg"
    if not cat.is_file():
        pytest.skip(f"fixture asset missing: {cat}")

    files = {"image": (cat.name, cat.read_bytes())}
    r = client.post("/search", data={"q": "", "limit": "1"}, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    top = body["results"][0]
    assert Path(top["file_paths"][0]).name == "cat-portrait.jpg"
    assert top["similarity"] > 0.999, top["similarity"]
