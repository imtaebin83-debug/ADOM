from pathlib import Path
from collections import Counter

root = Path(r"C:\Users\gahyu\RUGD\raw")

extensions = Counter(
    path.suffix.lower()
    for path in root.rglob("*")
    if path.is_file()
)

print("검색 경로:", root)
print("폴더 존재 여부:", root.exists())
print("확장자:", extensions)