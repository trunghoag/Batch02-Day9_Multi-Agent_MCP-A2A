"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma túy 2021 (73/2021/QH14)",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
        "filename": "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy.pdf",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy-va-tien-chat.pdf",
        "title": "Nghị định 57/2022/NĐ-CP về danh mục chất ma túy và tiền chất",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Tải một văn bản pháp luật gốc về data/landing/legal/."""
    setup_directory()
    filepath = DATA_DIR / filename
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath} ({filepath.stat().st_size:,} bytes)")
    return filepath


def collect_legal_documents() -> list[Path]:
    """Tải tối thiểu 3 văn bản pháp luật theo yêu cầu Task 1."""
    downloaded_files = []
    for document in LEGAL_DOCUMENTS:
        print(f"Đang tải: {document['title']}")
        downloaded_files.append(download_file(document["url"], document["filename"]))
    return downloaded_files


if __name__ == "__main__":
    collect_legal_documents()
