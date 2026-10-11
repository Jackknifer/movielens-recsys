import ast
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
PARSER = MarkdownIt("commonmark").enable("table")


def local_links(path):
    links = []
    for token in PARSER.parse(path.read_text(encoding="utf-8")):
        for child in token.children or []:
            if child.type == "link_open":
                target = urlsplit(child.attrGet("href"))
                if not target.scheme and target.path:
                    links.append((path.parent / unquote(target.path)).resolve())
    return links


def test_current_document_links_exist():
    documents = list(ROOT.glob("*.md"))
    assert ROOT / "README.md" in documents
    for document in documents:
        for target in local_links(document):
            assert target.exists(), f"{document.relative_to(ROOT)}：{target}"


def test_public_documents_link_only_public_files():
    private = {".local", ".venv", ".work", "data", "artifacts", "outputs"}
    for document in ROOT.glob("*.md"):
        for target in local_links(document):
            assert target.is_relative_to(ROOT)
            assert target.relative_to(ROOT).parts[0] not in private


def test_runtime_imports_use_current_modules():
    for path in (ROOT / "src").glob("**/*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "learning", path
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "learning" for alias in node.names), path


def test_project_metadata_and_lock_match():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "movielens-recsys"
    assert any(
        package["name"] == project["project"]["name"] and package.get("source", {}).get("virtual") == "."
        for package in lock["package"]
    )
