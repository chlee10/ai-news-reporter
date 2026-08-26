from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    region: str
    trust: float


DEFAULT_SOURCES = (
    Source("OpenAI", "https://openai.com/news/rss.xml", "global", 1.0),
    Source("Google AI", "https://blog.google/technology/ai/rss/", "global", 1.0),
    Source("Microsoft AI", "https://blogs.microsoft.com/ai/feed/", "global", 0.95),
    Source("Anthropic", "https://www.anthropic.com/rss.xml", "global", 1.0),
    Source("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "global", 0.8),
    Source("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/", "global", 0.9),
    Source("ZDNet Korea", "https://zdnet.co.kr/news_xml.asp?category=0400", "korea", 0.8),
    Source("AI타임스", "https://www.aitimes.com/rss/allArticle.xml", "korea", 0.7),
)