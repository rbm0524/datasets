import requests
from bs4 import BeautifulSoup
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import urllib3
import os
import re

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 저장 경로 설정
OUTPUT_DIR = "cambridge_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selenium 설정
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors=yes')
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # 자동화 감지 방지
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])  # 자동화 감지 방지
chrome_options.add_experimental_option('useAutomationExtension', False)  # 자동화 감지 방지

# 최신 방식으로 드라이버 초기화 (옵션)
try:
    driver = webdriver.Chrome(options=chrome_options)
except Exception as e:
    print(f"기본 방식으로 Chrome 초기화 실패: {e}")
    try:
        # 대체 방법으로 시도
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"대체 방식으로도 Chrome 초기화 실패: {e}")
        driver = webdriver.Chrome(options=chrome_options)  # 마지막 시도

# 자동화 감지 방지를 위한 추가 설정
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# 각 URL에서 문법 페이지 직접 크롤링
def crawl_grammar_page(url):
    try:
        # 페이지 로딩
        driver.get(url)
        time.sleep(3)  # 충분한 로딩 시간
        
        # 페이지 HTML 저장 (디버깅용)
        page_name = url.split('/')[-1]
        if not page_name:
            page_name = "index"
        html_path = os.path.join(OUTPUT_DIR, f"{page_name}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"HTML 저장됨: {html_path}")
        
        # 현재 페이지 분석
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # 제목 찾기 - 다양한 방법 시도
        title = None
        
        # 1. 일반적인 제목 요소
        title_elements = soup.select(".cdo-section-title, h1, .headword, .di-title")
        for element in title_elements:
            if element.text.strip():
                title = element.text.strip()
                break
                
        # 2. 제목을 찾지 못한 경우 URL에서 추출
        if not title:
            title = url.split("/")[-1].replace("-", " ").title()
            if not title:
                title = "Cambridge Grammar"
        
        # 콘텐츠 추출 - 먼저 특정 클래스로 시도
        content = ""
        
        # 콘텐츠 찾기 시도 1: 클래스 기반
        content_sections = soup.select(".cdo-section-body, .entry-body__el, .entry-body, .ddef_block")
        for section in content_sections:
            if section.text.strip():
                # 단락만 추출하기
                paragraphs = section.select("p")
                if paragraphs:
                    content = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
                
                # 단락이 없으면 div 전체 텍스트
                if not content:
                    content = section.text.strip()
                
                # 충분한 내용이 있으면 중단
                if len(content) > 50:
                    break
        
        # 콘텐츠 찾기 실패: 일반 문단 추출 시도
        if not content or len(content) < 50:
            paragraphs = soup.select("p")
            content = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip() and len(p.text.strip()) > 30])
        
        # 콘텐츠 정리 (필요시)
        content = re.sub(r'\s+', ' ', content)  # 여러 공백을 하나로
        content = re.sub(r'\n\s*\n', '\n\n', content)  # 빈 줄 정리
        
        # 관련 문법 페이지 링크 찾기
        links = []
        for a in soup.select("a"):
            href = a.get("href", "")
            if href and "/grammar/british-grammar/" in href:
                link_url = "https://dictionary.cambridge.org" + href if href.startswith("/") else href
                links.append({
                    "title": a.text.strip() or href.split("/")[-1].replace("-", " ").title(),
                    "url": link_url
                })
        
        # 결과 생성
        if content and len(content) > 100:  # 최소 콘텐츠 길이 확인
            return {
                "id": f"cambridge_{title[:40].replace(' ', '_').replace('/', '_')}",
                "source": "Cambridge Grammar",
                "title": title,
                "content": content,
                "url": url,
                "related_links": links[:10]  # 최대 10개 관련 링크
            }
        else:
            print(f"콘텐츠가 충분하지 않음: {url}")
            return None
            
    except Exception as e:
        print(f"페이지 크롤링 중 오류: {url} - {e}")
        return None

# 문법 카테고리 목록
GRAMMAR_CATEGORIES = [
    "adjectives-and-adverbs",
    "clauses-and-sentences",
    "determiners-and-quantifiers",
    "ellipsis",
    "function-words",
    "idioms", 
    "modality",
    "modifiers-in-nounal-groups",
    "negation",
    "nouns",
    "past",
    "phrasal-verbs",
    "pronouns",
    "questions",
    "relative-clauses",
    "reported-speech",
    "verbs"
]

# 실행부
if __name__ == "__main__":
    try:
        print("Cambridge 문법 콘텐츠 크롤링 시작...")
        all_docs = []
        
        # 메인 문법 페이지부터 시작
        base_url = "https://dictionary.cambridge.org/grammar/british-grammar/"
        print(f"메인 페이지 크롤링: {base_url}")
        
        main_page = crawl_grammar_page(base_url)
        if main_page:
            all_docs.append(main_page)
            print(f"메인 페이지에서 {len(main_page['related_links'])}개 링크 발견")
            
            # 메인 페이지에서 발견한 링크들 크롤링
            for link in main_page['related_links']:
                print(f"링크 크롤링: {link['title']} ({link['url']})")
                page_doc = crawl_grammar_page(link['url'])
                if page_doc:
                    all_docs.append(page_doc)
                    print(f"✓ 성공: {page_doc['title']} (내용 길이: {len(page_doc['content'])})")
                time.sleep(1.5)  # 요청 간격 두기
        
        # 직접 카테고리 URL도 시도
        print("\n카테고리 직접 크롤링 시작...")
        for category in GRAMMAR_CATEGORIES:
            category_url = f"https://dictionary.cambridge.org/grammar/british-grammar/{category}"
            print(f"카테고리 크롤링: {category}")
            
            category_doc = crawl_grammar_page(category_url)
            if category_doc:
                all_docs.append(category_doc)
                print(f"✓ 카테고리 성공: {category_doc['title']}")
                
                # 카테고리에서 발견한 링크들 중 일부만 크롤링 (최대 5개)
                links_to_crawl = category_doc['related_links'][:5]
                for link in links_to_crawl:
                    print(f"  하위 링크 크롤링: {link['title']}")
                    sub_doc = crawl_grammar_page(link['url'])
                    if sub_doc:
                        all_docs.append(sub_doc)
                        print(f"  ✓ 성공: {sub_doc['title']}")
                    time.sleep(1.5)  # 요청 간격 두기
            
            time.sleep(2)  # 카테고리 간 간격
        
        # 중복 제거 (URL 기준)
        unique_docs = []
        visited_urls = set()
        for doc in all_docs:
            if doc['url'] not in visited_urls:
                visited_urls.add(doc['url'])
                unique_docs.append(doc)
                
        print(f"\nCambridge 문법 콘텐츠 수집 완료: {len(unique_docs)}개 고유 문서")
        
        # 결과 저장
        with open("cambridge_grammar_documents.json", "w", encoding="utf-8") as f:
            json.dump(unique_docs, f, ensure_ascii=False, indent=2)
            
        print("문서 저장 완료 → cambridge_grammar_documents.json")
        
    except Exception as e:
        print(f"메인 실행 중 오류 발생: {e}")
    finally:
        # 브라우저 종료
        driver.quit()