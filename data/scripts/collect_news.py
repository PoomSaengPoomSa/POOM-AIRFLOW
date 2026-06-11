"""
SBS 뉴스 RSS → Elasticsearch 적재
카테고리: 정치(01), 경제(02), 국제(07)
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch, helpers
from datetime import datetime
import hashlib
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

import os
# ── 설정 ──────────────────────────────────────────────────────────────────────

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "sbs_news"
ES_USER = None          # 인증 필요 시 설정
ES_PASSWORD = None

RSS_FEEDS = {
    "정치": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER",
    "경제": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=02&plink=RSSREADER",
    "국제": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=07&plink=RSSREADER",
}

REQUEST_DELAY = 1.0     # 기사 페이지 크롤링 간격 (초)
REQUEST_TIMEOUT = 10    # HTTP 타임아웃 (초)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Elasticsearch 인덱스 매핑 ─────────────────────────────────────────────────

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id":       {"type": "keyword"},
            "category":     {"type": "keyword"},
            "title":        {"type": "text", "analyzer": "standard"},
            "content":      {"type": "text", "analyzer": "standard"},
            "summary":      {"type": "text", "analyzer": "standard"},
            "url":          {"type": "keyword"},
            "published_at": {"type": "date"},
            "indexed_at":   {"type": "date"},
            "author":       {"type": "keyword"},
        }
    }
    # settings 블록 전체 삭제
}

# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def make_doc_id(url: str) -> str:
    """URL 기반 고유 ID 생성 (중복 방지용)"""
    return hashlib.md5(url.encode()).hexdigest()


def parse_published_date(entry) -> str | None:
    """feedparser의 published_parsed → ISO 8601 및 KST(UTC+9) 변환"""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        from datetime import timezone, timedelta
        utc_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        kst_tz = timezone(timedelta(hours=9))
        kst_dt = utc_dt.astimezone(kst_tz)
        return kst_dt.isoformat()
    return None


# ── 기사 전문 크롤링 ──────────────────────────────────────────────────────────

def fetch_article_content(url: str) -> str:
    """
    SBS 뉴스 기사 페이지에서 본문 텍스트 추출.
    선택자는 실제 HTML 구조에 따라 조정 필요.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"기사 페이지 요청 실패: {url} → {e}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    # SBS 뉴스 본문 컨테이너 후보 선택자 (우선순위 순)
    selectors = [
        "div.text_area",        # 주요 본문 영역
        "div#news_content",
        "article.article_view",
        "div.article_body",
    ]

    for sel in selectors:
        container = soup.select_one(sel)
        if container:
            # 불필요 태그 제거
            for tag in container.select("script, style, figure, .ad_area, .related_news"):
                tag.decompose()
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 50:  # 너무 짧으면 다음 선택자 시도
                return text

    logger.warning(f"본문 추출 실패 (선택자 불일치): {url}")
    return ""


# ── RSS 파싱 ──────────────────────────────────────────────────────────────────

def parse_rss(category: str, feed_url: str) -> list[dict]:
    """RSS 피드 파싱 + 각 기사 전문 크롤링 → 문서 리스트 반환"""
    feed = feedparser.parse(feed_url)

    if feed.bozo:
        logger.error(f"[{category}] RSS 파싱 오류: {feed.bozo_exception}")
        return []

    logger.info(f"[{category}] 항목 수: {len(feed.entries)}")
    docs = []

    for entry in feed.entries:
        url = entry.get("link", "")
        if not url:
            continue

        doc_id = make_doc_id(url)
        title = entry.get("title", "").strip()
        summary = BeautifulSoup(entry.get("summary", ""), "lxml").get_text(strip=True)
        published_at = parse_published_date(entry)
        author = entry.get("author", "")

        logger.info(f"  크롤링: {title[:40]}...")
        content = fetch_article_content(url)
        time.sleep(REQUEST_DELAY)

        docs.append({
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": {
                "doc_id":       doc_id,
                "category":     category,
                "title":        title,
                "content":      content,
                "summary":      summary,
                "url":          url,
                "published_at": published_at,
                "indexed_at":   datetime.utcnow().isoformat(),
                "author":       author,
            }
        })

    return docs


# ── Elasticsearch ─────────────────────────────────────────────────────────────

def get_es_client() -> Elasticsearch:
    kwargs = {"hosts": [ES_HOST]}
    if ES_USER and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)
    return Elasticsearch(**kwargs)


def ensure_index(es: Elasticsearch) -> None:
    try:
        es.indices.get(index=ES_INDEX)
        logger.info(f"인덱스 기존 존재: {ES_INDEX}")
    except Exception:
        es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
        logger.info(f"인덱스 생성: {ES_INDEX}")


def bulk_index(es: Elasticsearch, docs: list[dict]) -> tuple[int, int]:
    """bulk API로 적재. (성공 수, 실패 수) 반환"""
    if not docs:
        return 0, 0

    success, errors = helpers.bulk(
        es,
        docs,
        raise_on_error=False,
        stats_only=False,
    )
    failed = len(errors) if isinstance(errors, list) else 0
    return success, failed


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    es = get_es_client()

    try:
        es.ping()
        logger.info(f"ES 연결 성공: {ES_HOST}")
    except Exception as e:
        logger.error(f"ES 연결 실패: {e}")
        return

    ensure_index(es)

    total_success = 0
    total_failed = 0

    for category, feed_url in RSS_FEEDS.items():
        logger.info(f"=== [{category}] RSS 처리 시작 ===")
        docs = parse_rss(category, feed_url)
        success, failed = bulk_index(es, docs)
        logger.info(f"[{category}] 적재 완료 → 성공: {success}, 실패: {failed}")
        total_success += success
        total_failed += failed

    logger.info(f"전체 완료 → 성공: {total_success}, 실패: {total_failed}")


if __name__ == "__main__":
    main()