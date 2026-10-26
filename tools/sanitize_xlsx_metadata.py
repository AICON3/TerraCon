"""Remove personal author and edit-time metadata from XLSX package properties.

This utility leaves worksheets, formulas, and shared strings unchanged. It is
not a guarantee of anonymity: review data values, images, and git history too.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile
from xml.etree import ElementTree as ET

CORE = "docProps/core.xml"
NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}
REMOVE = {
    f"{{{NS['dc']}}}creator",
    f"{{{NS['cp']}}}lastModifiedBy",
    f"{{{NS['dcterms']}}}created",
    f"{{{NS['dcterms']}}}modified",
}


def scrub(path: Path) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as src:
        if CORE not in src.namelist():
            return False
        root = ET.fromstring(src.read(CORE))
        removed = False
        for child in list(root):
            if child.tag in REMOVE:
                root.remove(child)
                removed = True
        if not removed:
            return False
        core = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            with ZipFile(temp_path, "w") as dst:
                for info in src.infolist():
                    dst.writestr(info, core if info.filename == CORE else src.read(info.filename))
            with ZipFile(temp_path) as check:
                assert check.testzip() is None
                assert not any(x.tag in REMOVE for x in ET.fromstring(check.read(CORE)))
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for filename in args.files:
        changed = scrub(filename)
        print(f"{filename}: {'metadata sanitized' if changed else 'no author metadata found'}")


if __name__ == "__main__":
    main()
