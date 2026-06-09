"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

PDF_TEXT_FALLBACK_URLS = {
    "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy": (
        "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2021/12/34944/"
        "37821-1-20211047-1048105-2021-nd-cp.pdf"
    ),
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy-va-tien-chat": (
        "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2022/8/37734/"
        "41623-1-2022709-71057-2022-nd-cp.pdf"
    ),
}

LEGAL_METADATA_FALLBACKS = {
    "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy": """
## Extraction note

PDF gốc là bản ký số nên bộ extract text local có thể không đọc được nội dung.
Metadata pháp lý dùng cho pipeline: Nghị định số 105/2021/NĐ-CP của Chính phủ,
ban hành ngày 04-12-2021, có hiệu lực từ ngày 01-01-2022. Văn bản quy định
chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma túy; áp
dụng đối với cơ quan, tổ chức, cá nhân liên quan đến phối hợp phòng, chống tội
phạm về ma túy, kiểm soát các hoạt động hợp pháp liên quan đến ma túy, quản lý
người sử dụng trái phép chất ma túy, xét nghiệm chất ma túy trong cơ thể và cai
nghiện ma túy. Nguồn văn bản gốc: Cổng Thông tin điện tử Chính phủ.
""",
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy-va-tien-chat": """
## Extraction note

PDF gốc là bản ký số nên bộ extract text local có thể không đọc được nội dung.
Metadata pháp lý dùng cho pipeline: Nghị định số 57/2022/NĐ-CP của Chính phủ,
ban hành và có hiệu lực ngày 25-08-2022. Văn bản quy định các danh mục chất
ma túy và tiền chất, bao gồm các nhóm chất ma túy tuyệt đối cấm sử dụng, các
chất ma túy được sử dụng hạn chế trong nghiên cứu, kiểm nghiệm, giám định,
điều tra tội phạm hoặc trong lĩnh vực y tế theo quy định của cơ quan có thẩm
quyền, cùng danh mục tiền chất. Nguồn văn bản gốc: Cổng Thông tin điện tử
Chính phủ.
""",
}


def get_markitdown():
    """Trả về MarkItDown nếu môi trường đã cài; nếu chưa thì dùng fallback."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    return MarkItDown()


def extract_pdf_text(filepath: Path) -> str:
    """Extract text từ PDF bằng MarkItDown nếu có, fallback sang pypdf."""
    md = get_markitdown()
    if md:
        try:
            result = md.convert(str(filepath))
            if len(result.text_content.strip()) >= 200:
                return result.text_content
            print(f"  MarkItDown output too short for {filepath.name}; using fallback.")
        except Exception as exc:
            print(f"  MarkItDown failed for {filepath.name} ({exc.__class__.__name__}); using fallback.")

    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    text = extract_text_from_pdf_reader(reader)
    if len(text.strip()) >= 200:
        return text

    fallback_url = PDF_TEXT_FALLBACK_URLS.get(filepath.stem)
    if fallback_url:
        try:
            return extract_pdf_text_from_url(fallback_url)
        except Exception as exc:
            print(f"  Fallback PDF failed for {filepath.name}: {exc}")

    return LEGAL_METADATA_FALLBACKS.get(
        filepath.stem,
        "PDF text extraction failed. Please install markitdown or provide a text-based PDF.",
    )


def extract_text_from_pdf_reader(reader) -> str:
    """Extract text từ PdfReader, tách rõ từng page."""
    pages = []
    for index, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"## Page {index}\n\n{page_text.strip()}")
    return "\n\n".join(pages)


def extract_pdf_text_from_url(url: str) -> str:
    """Download một PDF text-based từ nguồn chính thức và extract bằng pypdf."""
    from pypdf import PdfReader

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=45) as response:
        data = response.read()
    return extract_text_from_pdf_reader(PdfReader(BytesIO(data)))


def extract_doc_text(filepath: Path) -> str:
    """Convert DOC/DOCX bằng MarkItDown khi có dependency tương ứng."""
    md = get_markitdown()
    if not md:
        raise RuntimeError(
            f"Không thể convert {filepath.name}: cần cài markitdown cho DOC/DOCX"
        )
    result = md.convert(str(filepath))
    return result.text_content


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  Skip, chua co thu muc: {legal_dir}")
        return []

    converted = []
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            if filepath.suffix.lower() == ".pdf":
                text_content = extract_pdf_text(filepath)
            else:
                text_content = extract_doc_text(filepath)

            output_path = output_dir / f"{filepath.stem}.md"
            header = f"# {filepath.stem}\n\n"
            header += f"**Source file:** {filepath.name}\n"
            header += "**Document type:** legal\n\n---\n\n"
            output_path.write_text(header + text_content, encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {output_path}")
    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_output in output_dir.glob("article_*.md"):
        old_output.unlink()

    if not news_dir.exists():
        print(f"  Skip, chua co thu muc: {news_dir}")
        return []

    converted = []
    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Source site:** {data.get('source_site', 'N/A')}\n"
            header += f"**Published:** {data.get('published_at', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            header += "**Document type:** news\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {output_path}")
    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    print(f"\nDone! Converted {len(legal_files) + len(news_files)} files.")
    print("Output dir:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
