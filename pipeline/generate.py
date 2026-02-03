# pipeline/generate.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple
from slugify import slugify
import json
import random
import re
import html as _html
import urllib.request
from urllib.parse import urlparse

from pipeline.util import (
    ROOT,
    read_text,
    write_text,
    read_json,
    write_json,
    normalize_url,
    simple_tokens,
    jaccard,
    sanitize_llm_html,
)
from pipeline.deepseek import DeepSeekClient
from pipeline.reddit import fetch_rss_entries
from pipeline.render import env_for, render_to_file, write_asset

# -----------------------------
# Paths
# -----------------------------
# 販売テンプレとして「分かりやすさ優先」：configはレポ直下
CONFIG_PATH = ROOT / "config.json"
ADS_JSON_PATH = ROOT / "ads.json"

PROCESSED_PATH = ROOT / "processed_urls.txt"
ARTICLES_PATH = ROOT / "data" / "articles.json"
LAST_RUN_PATH = ROOT / "data" / "last_run.json"

TEMPLATES_DIR = ROOT / "pipeline" / "templates"
STATIC_DIR = ROOT / "pipeline" / "static"


# -----------------------------
# Default Ads (optional)
# NOTE: 商品としてはads.jsonを空テンプレにして、ここは空でもいい。
# -----------------------------
ADS_TOP = ""
ADS_MID = ""
ADS_BOTTOM = ""
ADS_RAIL_LEFT = ""
ADS_RAIL_RIGHT = ""

FIXED_POLICY_BLOCK = """
<p><strong>Policy & Transparency</strong></p>
<ul>
  <li><strong>Source & attribution:</strong> Each post is based on public RSS items. We link to the original source.</li>
  <li><strong>Original value:</strong> We add commentary, context, and takeaways. If uncertain: "Not stated in the source."</li>
  <li><strong>No manipulation:</strong> No cloaking, hidden text, doorway pages, or misleading metadata.</li>
  <li><strong>Safety filters:</strong> We skip obvious adult/self-harm/gore keywords and avoid NSFW feeds.</li>
  <li><strong>Ads:</strong> Third-party scripts may show ads we do not directly control.</li>
  <li><strong>Removal requests:</strong> If content should be removed, contact us with the URL and reason.</li>
</ul>
<p>Contact: <a href="mailto:{contact_email}">{contact_email}</a></p>
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.json not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_site_dir(cfg: dict) -> Path:
    """
    Output directory for the generated static site.
    Recommended for GitHub Pages: docs/
    """
    site_dir_str = (
        cfg.get("site", {}).get("site_dir")
        or cfg.get("output", {}).get("site_dir")
        or "docs"
    )
    return ROOT / str(site_dir_str).strip().lstrip("./")


def load_processed() -> set[str]:
    s = read_text(PROCESSED_PATH)
    lines = [normalize_url(x) for x in s.splitlines() if x.strip()]
    return set(lines)


def append_processed(url: str) -> None:
    url = normalize_url(url)
    existing = load_processed()
    if url in existing:
        return

    current = read_text(PROCESSED_PATH).rstrip()
    if current.strip():
        current += "\n"
    current += url + "\n"
    write_text(PROCESSED_PATH, current)


def is_blocked(title: str, blocked_kw: list[str]) -> bool:
    t = (title or "").lower()
    for kw in blocked_kw:
        if kw.lower() in t:
            return True
    return False


def compute_rankings(articles: list[dict]) -> list[dict]:
    return sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)


def related_articles(current: dict, articles: list[dict], k: int = 6) -> list[dict]:
    cur_tok = simple_tokens(current.get("title", ""))
    scored: list[tuple[float, dict]] = []
    for a in articles:
        if a.get("id") == current.get("id"):
            continue
        sim = jaccard(cur_tok, simple_tokens(a.get("title", "")))
        scored.append((sim, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for s, a in scored[:k] if s > 0.05]


# -----------------------------
# Ads / Affiliate (optional)
# -----------------------------
def load_ads_catalog() -> dict:
    if not ADS_JSON_PATH.exists():
        return {}
    try:
        return json.loads(ADS_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load ads.json: {e}")


def classify_genre(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    rules = [
        ("health", ["health", "hair", "sleep", "diet", "doctor", "study says", "medical", "wellness"]),
        ("beauty", ["skincare", "beauty", "cosmetic", "laser", "dermatology", "makeup"]),
        ("finance", ["stock", "crypto", "bitcoin", "bank", "interest rate", "loan", "tax", "investment"]),
        ("tech", ["ai", "openai", "model", "gpu", "software", "bug", "security", "iphone", "android"]),
        ("travel", ["travel", "flight", "hotel", "trip", "tourism", "airport"]),
        ("study", ["learn", "exam", "toeic", "eiken", "study", "university"]),
        ("vpn", ["vpn", "privacy", "proxy", "geoblock"]),
        ("tools", ["tool", "formatter", "converter", "generator", "app", "extension"]),
    ]
    for genre, kws in rules:
        if any(k in text for k in kws):
            return genre
    return "general"


RELATED_GENRES: dict[str, list[str]] = {
    "tech": ["tools", "study", "finance"],
    "finance": ["tech", "tools"],
    "health": ["beauty", "tools"],
    "beauty": ["health", "tools"],
    "travel": ["tools", "finance"],
    "study": ["tools", "tech"],
}


def choose_ad(ads_catalog: dict, genre: str) -> Tuple[Optional[dict], Optional[str]]:
    if not isinstance(ads_catalog, dict):
        return (None, None)

    def pool_for(g: str) -> list[dict]:
        v = ads_catalog.get(g) or []
        if not isinstance(v, list):
            return []
        return [x for x in v if isinstance(x, dict) and str(x.get("code", "")).strip()]

    if genre != "general":
        pool = pool_for(genre)
        if pool:
            return (random.choice(pool), genre)

    for g in RELATED_GENRES.get(genre, []):
        pool = pool_for(g)
        if pool:
            return (random.choice(pool), g)

    all_ads: list[dict] = []
    for k, v in ads_catalog.items():
        if k == "general":
            continue
        if isinstance(v, list):
            all_ads.extend([x for x in v if isinstance(x, dict) and str(x.get("code", "")).strip()])
    if not all_ads:
        return (None, None)
    return (random.choice(all_ads), None)


def build_affiliate_section(
    cfg: dict,
    article_id: str,
    title: str,
    summary: str,
    ads_catalog: dict,
) -> Tuple[str, Optional[str]]:
    genre = classify_genre(title, summary)
    ad, _picked = choose_ad(ads_catalog, genre)
    if not ad:
        return ("", None)

    ad_id = str(ad.get("id", "")).strip() or None
    raw_code = str(ad.get("code", "")).strip()
    if not raw_code:
        return ("", None)

    go_base_url = (cfg.get("site", {}).get("go_base_url") or "").strip().rstrip("/")
    tracked_url = ""
    if go_base_url and ad_id:
        tracked_url = f"{go_base_url}/go?ad={_html.escape(ad_id)}&a={_html.escape(article_id)}"

    block = f"""
<section class="card affiliate">
  <div class="card-h">
    <div><strong>Recommended</strong></div>
    <div class="muted">Category: {_html.escape(genre)}</div>
  </div>

  <div class="ad-slot ad-affiliate">
    {raw_code}
  </div>
  {"<div class='cta-row'><a class='pill' href='"+tracked_url+"' rel='nofollow sponsored noopener' target='_blank'>View offer</a></div>" if tracked_url else ""}
</section>
""".strip()

    return (block, ad_id)


# -----------------------------
# OG Image cache (optional)
# -----------------------------
def _guess_ext_from_url(u: str) -> str:
    try:
        path = urlparse(u).path.lower()
    except Exception:
        path = ""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def cache_og_image(cfg: dict, site_dir: Path, base_url: str, src_url: str, article_id: str) -> str:
    src_url = (src_url or "").strip()
    if not src_url:
        return ""

    ext = _guess_ext_from_url(src_url)
    rel = f"/og/{article_id}{ext}"
    out_path = site_dir / rel.lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        try:
            ua = (cfg.get("site", {}).get("user_agent") or "").strip()
            if not ua:
                ua = "Mozilla/5.0 (compatible; AutoSiteBot/1.0)"
            req = urllib.request.Request(src_url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if data:
                out_path.write_bytes(data)
        except Exception:
            return ""

    return base_url.rstrip("/") + rel


# -----------------------------
# Candidate picking
# -----------------------------
def pick_candidate(cfg: dict, processed: set[str], articles: list[dict]) -> Optional[dict]:
    safety = cfg.get("safety", {})
    feeds = cfg.get("feeds", {})

    blocked_kw = safety.get("blocked_keywords", []) or []
    rss_list = feeds.get("reddit_rss", []) or []
    if not rss_list:
        raise RuntimeError("config.feeds.reddit_rss is empty. Add at least one RSS URL.")

    prev_titles = [a.get("title", "") for a in articles]
    prev_tok = [simple_tokens(t) for t in prev_titles if t]

    candidates: list[dict] = []
    for rss in rss_list:
        for e in fetch_rss_entries(rss):
            link = normalize_url(e.get("link", ""))
            title = e.get("title", "")
            if not link or link in processed:
                continue
            if is_blocked(title, blocked_kw):
                continue

            tok = simple_tokens(title)
            too_similar = any(jaccard(tok, pt) >= 0.78 for pt in prev_tok)
            if too_similar:
                continue

            e["image_url"] = e.get("hero_image", "") or ""
            e["image_kind"] = e.get("hero_image_kind", "none") or "none"
            candidates.append(e)

    if not candidates:
        return None

    if cfg.get("generation", {}).get("pick_random", False):
        return random.choice(candidates)
    return candidates[0]


# -----------------------------
# DeepSeek generation
# -----------------------------
def deepseek_article(cfg: dict, item: dict) -> Tuple[str, str]:
    ds = DeepSeekClient()

    gen = cfg.get("generation", {})
    model = gen.get("model", "deepseek-chat")
    temp = float(gen.get("temperature", 0.7))
    max_tokens = int(gen.get("max_tokens", 2400))

    title = item.get("title", "").strip()
    link = item.get("link", "").strip()
    summary = (item.get("summary", "") or "").strip()

    ads_catalog = {}
    ad = None
    if cfg.get("ads", {}).get("enable_ad_context_in_prompt", True):
        ads_catalog = load_ads_catalog()
        genre = classify_genre(title, summary)
        ad, _ = choose_ad(ads_catalog, genre)

    ad_genre = classify_genre(title, summary)
    ad_title = (ad.get("title") if ad else "") or ""
    ad_detail = (ad.get("detail") if ad else "") or ""

    system = (
        "You are a high-performance conversion copywriter and tech analyst. "
        "Write in English only. Do not fabricate facts. "
        "Be punchy and direct, but remain ethically grounded. "
        "If something is not stated in the source, say: 'Not stated in the source.'"
    )

    user = f"""
OUTPUT RULES:
- First line MUST be: TITLE: <your best SEO-friendly title>
- Second line MUST be empty.
- From the third line, output the HTML body only.
- Allowed tags: <p>, <h2>, <ul>, <li>, <strong>, <code>, <a>
- Do NOT output <h1>.
- Do NOT paste affiliate code or scripts.

INPUT:
Post title: {title}
Permalink: {link}
RSS summary snippet: {summary}

If irrelevant/low-value:
- Start with: <p><strong>[SKIP: no actionable value]</strong></p>
- One short <p> explaining why. Stop.

STRUCTURE:
1) <p><strong>[CRITICAL SUMMARY]</strong>: 2 lines. Who is at risk and the urgent action.</p>
2) <h2>Is this your problem?</h2> + 5 yes/no bullets
3) <h2>The Hidden Reality</h2> (impact-focused)
4) <h2>Stop the Damage / Secure the Win</h2> (3–7 steps)
5) <h2>The High Cost of Doing Nothing</h2>
6) <h2>Common Misconceptions</h2> (3–5 bullets)
7) <h2>Critical FAQ</h2> (5 bullets; if missing: Not stated in the source.)
8) <h2>Verify Original Details</h2> link to source
9) <h2>Strategic Next Step</h2>
   End with: "If you want a practical option people often use to handle this, here’s one."

AD CONTEXT (neutral; do NOT sell):
- Genre: {ad_genre}
- Ad title: {ad_title}
- Ad detail: {ad_detail}
""".strip()

    out = ds.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )

    out = (out or "").strip()

    m = re.match(r"(?is)^\s*TITLE:\s*(.+?)\s*\n\s*\n(.*)$", out)
    if m:
        llm_title = m.group(1).strip()
        llm_html = m.group(2).strip()
    else:
        llm_title = title
        llm_html = out

    llm_html = sanitize_llm_html(llm_html or "")
    return (llm_title, llm_html)


def strip_leading_duplicate_title(body_html: str, title: str) -> str:
    if not body_html or not title:
        return body_html

    t = _html.unescape(title).strip()
    t_norm = re.sub(r"\s+", " ", t).lower()

    def _same(text: str) -> bool:
        x = _html.unescape(text or "").strip()
        x = re.sub(r"\s+", " ", x).lower()
        return x == t_norm

    s = body_html.lstrip()

    m = re.match(r"(?is)^\s*<h1[^>]*>(.*?)</h1>\s*", s)
    if m and _same(m.group(1)):
        return s[m.end():].lstrip()

    m = re.match(r"(?is)^\s*<h2[^>]*>(.*?)</h2>\s*", s)
    if m and _same(m.group(1)):
        return s[m.end():].lstrip()

    m = re.match(r"(?is)^\s*<p[^>]*>(.*?)</p>\s*", s)
    if m:
        inner = re.sub(r"(?is)<[^>]+>", "", m.group(1))
        if _same(inner):
            return s[m.end():].lstrip()

    return body_html


# -----------------------------
# Site build
# -----------------------------
def write_rss_feed(cfg: dict, site_dir: Path, articles: list[dict], limit: int = 10) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    site_title = cfg["site"].get("title", "AutoSite")
    site_desc = cfg["site"].get("description", "Daily digest")

    items = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:limit]

    def rfc822(iso: str) -> str:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append("<rss version='2.0' xmlns:atom='http://www.w3.org/2005/Atom'>")
    parts.append("<channel>")
    parts.append(f"<title>{_html.escape(site_title)}</title>")
    parts.append(f"<link>{_html.escape(base_url + '/')}</link>")
    parts.append(f"<description>{_html.escape(site_desc)}</description>")
    parts.append(f"<lastBuildDate>{_html.escape(rfc822(now_utc_iso()))}</lastBuildDate>")

    for a in items:
        url = f"{base_url}{a['path']}"
        title = a.get("title", "")
        pub = a.get("published_ts", now_utc_iso())
        summary = a.get("summary", "") or ""
        if not summary:
            summary = re.sub(r"\s+", " ", re.sub(r"(?is)<[^>]+>", " ", a.get("body_html", ""))).strip()[:240]

        parts.append("<item>")
        parts.append(f"<title>{_html.escape(title)}</title>")
        parts.append(f"<link>{_html.escape(url)}</link>")
        parts.append(f"<guid isPermaLink='true'>{_html.escape(url)}</guid>")
        parts.append(f"<pubDate>{_html.escape(rfc822(pub))}</pubDate>")
        parts.append(f"<description>{_html.escape(summary)}</description>")
        parts.append("</item>")

    parts.append("</channel>")
    parts.append("</rss>")

    (site_dir / "feed.xml").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_site(cfg: dict, site_dir: Path, articles: list[dict]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")

    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "articles").mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)

    write_asset(site_dir / "assets" / "style.css", STATIC_DIR / "style.css")
    if (STATIC_DIR / "fx.js").exists():
        write_asset(site_dir / "assets" / "fx.js", STATIC_DIR / "fx.js")

    robots = f"""User-agent: *
Allow: /

Sitemap: {base_url}/sitemap.xml
"""
    (site_dir / "robots.txt").write_text(robots, encoding="utf-8")

    urls = [f"{base_url}/"] + [f"{base_url}{a['path']}" for a in articles]
    sitemap_items = "\n".join([f"<url><loc>{u}</loc></url>" for u in urls])
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemap_items}
</urlset>
"""
    (site_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    jenv = env_for(TEMPLATES_DIR)

    ranking = compute_rankings(articles)[:10]
    new_articles = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:10]

    write_rss_feed(cfg, site_dir, articles, limit=10)

    ads_cfg = cfg.get("ads", {})
    ads_enabled = bool(ads_cfg.get("enabled", False))

    base_ctx = {
        "site": cfg["site"],
        "ranking": ranking,
        "new_articles": new_articles,
        "ads_top": ADS_TOP if ads_enabled else "",
        "ads_mid": ADS_MID if ads_enabled else "",
        "ads_bottom": ADS_BOTTOM if ads_enabled else "",
        "ads_rail_left": ADS_RAIL_LEFT if ads_enabled else "",
        "ads_rail_right": ADS_RAIL_RIGHT if ads_enabled else "",
        "now_iso": now_utc_iso(),
    }

    ctx = dict(base_ctx)
    ctx.update(
        {
            "title": cfg["site"].get("title", "AutoSite"),
            "description": cfg["site"].get("description", "Daily digest"),
            "canonical": base_url + "/",
            "og_type": "website",
            "og_image": "",
        }
    )
    render_to_file(jenv, "index.html", ctx, site_dir / "index.html")

    static_pages = [
        ("about", "About", "<p>Daily digest from public sources with added context and takeaways.</p>"),
        ("privacy", "Privacy", "<p>We do not require accounts. Third-party scripts may collect device identifiers.</p>"),
        ("terms", "Terms", "<p>Use at your own risk. We do not guarantee outcomes.</p>"),
        ("disclaimer", "Disclaimer", "<p>Not affiliated with any source. Trademarks belong to their owners.</p>"),
        ("contact", "Contact", f"<p>Email: <a href='mailto:{cfg['site']['contact_email']}'>{cfg['site']['contact_email']}</a></p>"),
    ]
    for slug, page_title, body in static_pages:
        ctx = dict(base_ctx)
        ctx.update(
            {
                "page_title": page_title,
                "page_body": body,
                "title": page_title,
                "description": cfg["site"].get("description", "Daily digest"),
                "canonical": f"{base_url}/{slug}.html",
                "og_type": "website",
                "og_image": "",
            }
        )
        render_to_file(jenv, "static.html", ctx, site_dir / f"{slug}.html")

    og_cfg = cfg.get("og", {})
    og_cache_enabled = bool(og_cfg.get("cache_images", True))

    for a in articles:
        rel = related_articles(a, articles, k=6)
        src = a.get("hero_image", "") or ""
        og_img = ""
        if og_cache_enabled:
            og_img = cache_og_image(cfg, site_dir, base_url, src, a.get("id", "article"))

        ctx = dict(base_ctx)
        ctx.update(
            {
                "a": a,
                "related": rel,
                "policy_block": FIXED_POLICY_BLOCK.format(contact_email=cfg["site"]["contact_email"]),
                "title": a.get("title", cfg["site"].get("title", "AutoSite")),
                "description": (a.get("summary", "") or cfg["site"].get("description", "Daily digest"))[:200],
                "canonical": f"{base_url}{a['path']}",
                "og_type": "article",
                "og_image": og_img,
            }
        )
        render_to_file(jenv, "article.html", ctx, site_dir / a["path"].lstrip("/"))


def write_last_run(cfg: dict, payload: dict[str, Any]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    out = {"updated_utc": now_utc_iso(), "homepage_url": base_url + "/", **payload}
    write_json(LAST_RUN_PATH, out)


def main() -> None:
    cfg = load_config()

    base_url = (cfg.get("site", {}).get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError("config.site.base_url is missing (example: https://YOURNAME.github.io/YOURREPO)")

    site_dir = get_site_dir(cfg)

    processed = load_processed()
    articles = read_json(ARTICLES_PATH, default=[])

    cand = pick_candidate(cfg, processed, articles)
    if not cand:
        build_site(cfg, site_dir, articles)
        write_last_run(cfg, {"created": False, "article_url": "", "article_title": "", "source_url": "", "note": "No new candidate found. Site rebuilt."})
        return

    llm_title, body_html = deepseek_article(cfg, cand)
    body_html = strip_leading_duplicate_title(body_html, llm_title or cand.get("title", ""))

    affiliate_html = ""
    chosen_ad_id = None
    if cfg.get("ads", {}).get("enabled_affiliate_block", False):
        ads_catalog = load_ads_catalog()
        affiliate_html, chosen_ad_id = build_affiliate_section(
            cfg=cfg,
            article_id=f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{slugify(llm_title or cand.get('title',''))[:80] or 'post'}",
            title=llm_title or cand.get("title", ""),
            summary=cand.get("summary", "") or "",
            ads_catalog=ads_catalog,
        )
        print(f"[ads] chosen_ad_id={chosen_ad_id} affiliate_len={len(affiliate_html or '')}")

    if affiliate_html:
        body_html = body_html.rstrip() + "\n\n" + affiliate_html + "\n"

    ts = datetime.now(timezone.utc)
    ymd = ts.strftime("%Y-%m-%d")
    slug = slugify(llm_title or cand.get("title", ""))[:80] or f"post-{int(ts.timestamp())}"
    path = f"/articles/{ymd}-{slug}.html"
    article_url = base_url.rstrip("/") + path

    entry = {
        "id": f"{ymd}-{slug}",
        "title": llm_title or cand.get("title", ""),
        "path": path,
        "published_ts": ts.isoformat(timespec="seconds"),
        "source_url": cand.get("link", ""),
        "rss": cand.get("rss", ""),
        "summary": cand.get("summary", ""),
        "body_html": body_html,
        "hero_image": cand.get("image_url", "") or "",
        "hero_image_kind": cand.get("image_kind", "none") or "none",
    }

    append_processed(cand.get("link", ""))
    articles.insert(0, entry)
    write_json(ARTICLES_PATH, articles)

    build_site(cfg, site_dir, articles)

    write_last_run(cfg, {"created": True, "article_url": article_url, "article_path": path, "article_title": cand.get("title", ""), "source_url": cand.get("link", "")})


if __name__ == "__main__":
    main()
