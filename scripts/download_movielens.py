import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

from remotezip import RemoteZip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://funrec-datasets.s3.eu-west-3.amazonaws.com/funrec-movielens-1m.zip"
OFFICIAL_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED_FILES = ("users.pkl", "movies.pkl", "ratings.pkl")
EXPECTED_SHA256 = {
    "users.pkl": "1fe883dd0dcfaa974586cc98fff0e5c5cf5ae84d8b7abd86544f472c3a699525",
    "movies.pkl": "565921282d05f63e56f92f92a7ca6968447b78d80edd8bdfc5ea7ba122e57aaa",
    "ratings.pkl": "dd9db8c24f5889062aebba20b82402ba9c8529fe92b005956d4f5bbebb7a3fa9",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="获取原项目提供的 MovieLens 数据")
    parser.add_argument("--list", action="store_true", help="只查看需要的文件信息")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()
    target_dir = args.output_dir
    if not args.list:
        for name in (*REQUIRED_FILES, "ml-1m.zip", "source.json"):
            if (target_dir / name).exists():
                raise FileExistsError(target_dir / name)
        target_dir.mkdir(parents=True, exist_ok=True)

    # remotezip 负责 ZIP 解析及 HTTP 分段读取。
    with RemoteZip(SOURCE_URL, timeout=120) as archive:
        selected = {}
        for name in REQUIRED_FILES:
            matches = [
                info for info in archive.infolist()
                if Path(info.filename).name == name
                and "__MACOSX" not in Path(info.filename).parts
            ]
            if len(matches) != 1:
                raise ValueError(f"无法唯一确定数据文件: {name}")
            selected[name] = matches[0]
        if args.list:
            for name, info in selected.items():
                print(name, info.filename, info.file_size, info.compress_size)
            return

        records = []
        for name, info in selected.items():
            content = archive.read(info)
            digest = hashlib.sha256(content).hexdigest()
            if digest != EXPECTED_SHA256[name]:
                raise ValueError(f"来源数据的 SHA-256 不符：{name}")
            with (target_dir / name).open("xb") as destination:
                destination.write(content)
            records.append({
                "file": name,
                "archive_member": info.filename,
                "bytes": len(content),
                "sha256": digest,
            })
            print(f"已保存 {name}: {len(content)} bytes", flush=True)

    official_path = target_dir / "ml-1m.zip"
    with urlopen(OFFICIAL_URL, timeout=120) as source, official_path.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    with ZipFile(official_path) as archive:
        archive.getinfo("ml-1m/ratings.dat")
        invalid_member = archive.testzip()
        if invalid_member is not None:
            raise ValueError(f"官方 ZIP 校验失败：{invalid_member}")
    with official_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    records.append({
        "file": "ml-1m.zip", "source_url": OFFICIAL_URL,
        "bytes": official_path.stat().st_size, "sha256": digest,
    })
    print(f"已保存 ml-1m.zip: {official_path.stat().st_size} bytes", flush=True)
    manifest = {"source_url": SOURCE_URL, "official_reference_url": OFFICIAL_URL, "files": records}
    with (target_dir / "source.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
