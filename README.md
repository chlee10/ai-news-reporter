# AI News Reporter

매일 오전 8시, 정오, 오후 5시(Asia/Seoul)에 국내외 AI 뉴스를 수집해 중복 제거, 출처 검증, 중요도 평가, 리포트 생성, Gmail 발송까지 수행합니다. 각 실행의 품질 지표와 개선 제안은 SQLite에 저장되며 다음 리포트의 편집 원칙에 반영됩니다.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test,llm]"
Copy-Item .env.example .env
```

`.env`에 Gmail 주소, [Google App Password](https://myaccount.google.com/apppasswords), 수신자 주소를 입력합니다. 일반 Gmail 비밀번호는 사용할 수 없습니다. 해외 기사 한국어 번역에는 `GEMINI_API_KEY`를 설정합니다. Gemini 오류 시 `OPENAI_API_KEY`, 이어서 Google 번역으로 자동 대체합니다.

## Run

```powershell
# 수집부터 이메일 발송까지 한 번 실행
ai-news run

# 이메일 없이 HTML/텍스트 리포트만 확인
ai-news run --dry-run

# 이미 발송한 기사를 포함해 개선된 리포트를 한 번 다시 발송
ai-news run --force

# 매일 08:00, 12:00, 17:00 KST 실행 프로세스 시작
ai-news schedule
```

GitHub Actions나 Windows Task Scheduler 같은 외부 스케줄러를 사용할 경우 `ai-news run`을 매일 08:00, 12:00, 17:00 KST에 실행하면 됩니다. `schedule`은 항상 실행 중인 서버/PC용입니다.

## Clickable Web Report

Gmail은 이메일 안의 JavaScript와 접기/펼치기 제어를 차단합니다. `.github/workflows/daily-report.yml`은 동일한 리포트를 GitHub Pages에 배포하며, 이메일의 `상세 요약 보기` 링크는 브라우저의 실제 토글 페이지로 연결됩니다. GitHub에 저장소를 만든 뒤 Pages의 Source를 **GitHub Actions**로 설정하고, Actions secrets에 `GMAIL_USERNAME`, `GMAIL_APP_PASSWORD`, `REPORT_RECIPIENTS`, `GEMINI_API_KEY`를 등록하세요. `REPORT_PUBLIC_URL`은 `https://계정명.github.io/저장소명` 형식입니다.

## Pipeline

1. **수집**: 공식 블로그, 국내외 기술 매체 RSS를 읽습니다.
2. **분석**: 제목/요약을 정규화하고 주제(모델, 정책, 산업, 연구, 안전)를 분류합니다. 해외 기사는 Gemini API로 자연스러운 한국어로 번역하고, 원문 본문을 가져와 기사별 2~3문장 상세 요약을 만듭니다. Gemini 오류 시 OpenAI API, 이어서 Google 번역으로 자동 대체합니다.
3. **검증**: HTTPS 원문 URL, 허용 도메인, 중복 여부를 확인하고 출처 신뢰도와 교차 출처를 계산합니다.
4. **평가**: 신뢰도, 최신성, 교차 출처, AI 관련성으로 중요도를 점수화하고 국내 기사를 최대 4건 우선 포함합니다.
5. **개선**: 누락 URL, 출처 다양성, 상위 기사 품질을 평가해 저장하고 다음 실행 시 편집 가이드로 사용합니다.

소스 추가/수정은 `src/ai_news/sources.py` 또는 `.env`의 `EXTRA_FEEDS`에서 합니다.