"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_generated_articles():
    """Xóa output article_*.json cũ để lần crawl mới không tạo dữ liệu trùng."""
    setup_directory()
    for filepath in DATA_DIR.glob("article_*.json"):
        filepath.unlink()


ARTICLE_URLS = [
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-duoc-giam-2-nam-tu-3964578.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-bi-tinh-nghi-lien-quan-ma-tuy-4814289.html",
    "https://vietnamnet.vn/nha-thiet-ke-nguyen-cong-tri-bi-bat-lien-quan-duong-day-ma-tuy-2424939.html",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
]


class ArticleTextExtractor(HTMLParser):
    """HTML article extractor dùng thư viện chuẩn để không phụ thuộc Crawl4AI."""

    def __init__(self):
        super().__init__()
        self.meta = {}
        self.title_parts = []
        self.heading_parts = []
        self.description_parts = []
        self.paragraphs = []
        self._current_tag = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            content = attrs.get("content")
            if key and content:
                self.meta[key.lower()] = normalize_text(content)
        if tag in {"title", "h1", "h2", "p", "time"}:
            self._current_tag = tag

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == self._current_tag:
            self._current_tag = None

    def handle_data(self, data):
        if self._skip_depth or not self._current_tag:
            return
        text = normalize_text(data)
        if not text:
            return
        if self._current_tag == "title":
            self.title_parts.append(text)
        elif self._current_tag == "h1":
            self.heading_parts.append(text)
        elif self._current_tag == "h2":
            self.description_parts.append(text)
        elif self._current_tag == "p" and len(text) >= 30:
            self.paragraphs.append(text)
        elif self._current_tag == "time":
            self.meta.setdefault("published_time", text)


def normalize_text(text: str) -> str:
    """Chuẩn hóa whitespace và HTML entity."""
    return re.sub(r"\s+", " ", unescape(text)).strip()


def slugify(text: str, fallback: str) -> str:
    """Tạo tên file ASCII, ổn định."""
    replacements = {
        "đ": "d", "Đ": "d", "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
    }
    ascii_text = "".join(replacements.get(char, char) for char in text.lower())
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:90] or fallback


def fetch_html(url: str) -> str:
    """Download HTML với User-Agent giống trình duyệt để giảm khả năng bị chặn."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    crawl4ai_article = await crawl_article_with_crawl4ai(url)
    if crawl4ai_article:
        return crawl4ai_article

    html = await asyncio.to_thread(fetch_html, url)
    extractor = ArticleTextExtractor()
    extractor.feed(html)

    title = (
        extractor.meta.get("og:title")
        or extractor.meta.get("twitter:title")
        or " ".join(extractor.heading_parts[:1])
        or " ".join(extractor.title_parts[:1])
        or "Untitled"
    )
    title = normalize_text(title.split(" - ")[0])

    published_at = (
        extractor.meta.get("article:published_time")
        or extractor.meta.get("pubdate")
        or extractor.meta.get("published_time")
        or ""
    )
    description = (
        extractor.meta.get("og:description")
        or extractor.meta.get("description")
        or " ".join(extractor.description_parts[:1])
    )
    paragraphs = deduplicate_preserve_order(extractor.paragraphs)
    body = "\n\n".join(paragraphs)
    content_markdown = f"# {title}\n\n"
    if description:
        content_markdown += f"{normalize_text(description)}\n\n"
    content_markdown += body

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "published_at": normalize_text(published_at),
        "content_markdown": content_markdown.strip(),
        "source_site": url.split("/")[2],
        "crawler": "urllib-htmlparser",
    }


async def crawl_article_with_crawl4ai(url: str) -> dict | None:
    """
    Crawl bằng Crawl4AI theo README.

    Nếu môi trường chưa có browser runtime của Playwright/Crawl4AI thì trả None
    để fallback sang crawler nhẹ bằng thư viện chuẩn.
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        reason = exc.__class__.__name__
        print(f"  Crawl4AI unavailable ({reason}), fallback to urllib.")
        return None

    markdown = getattr(result, "markdown", "") or ""
    metadata = getattr(result, "metadata", {}) or {}
    title = normalize_text(
        metadata.get("title")
        or metadata.get("og:title")
        or extract_title_from_markdown(markdown)
        or "Untitled"
    )
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "published_at": normalize_text(
            metadata.get("article:published_time") or metadata.get("published_time") or ""
        ),
        "content_markdown": markdown.strip(),
        "source_site": url.split("/")[2],
        "crawler": "crawl4ai",
    }


def extract_title_from_markdown(markdown: str) -> str:
    """Lấy heading đầu tiên từ markdown Crawl4AI nếu metadata thiếu title."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    """Loại đoạn trùng, giữ thứ tự xuất hiện."""
    seen = set()
    result = []
    for item in items:
        normalized = normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    clean_generated_articles()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON với metadata rõ ràng.
        filename = f"article_{i:02d}_{slugify(article['title'], f'article-{i:02d}')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
