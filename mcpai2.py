import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from reportlab.pdfgen import canvas
import openai
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Load API key securely from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def crawl_website(url):
    """Extracts text, images, and meta info from the website."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    page_text = " ".join([p.text for p in soup.find_all("p")])
    images = [img["src"] for img in soup.find_all("img") if img.get("src")]

    return {
        "url": url,
        "text": page_text,
        "images": images,
        "title": soup.title.text if soup.title else "No Title",
        "meta_desc": soup.find("meta", attrs={"name": "description"}) or "No Meta Description",
        "h1_tags": [h1.text for h1 in soup.find_all("h1")],
        "alt_images": [img["src"] for img in soup.find_all("img") if not img.get("alt")],
        "canonical_url": soup.find("link", attrs={"rel": "canonical"})["href"] if soup.find("link", attrs={"rel": "canonical"}) else "No Canonical URL",
        "robots_txt": "Not Found" if requests.get(urljoin(url, "robots.txt")).status_code != 200 else "Found",
        "sitemap_xml": "Not Found" if requests.get(urljoin(url, "sitemap.xml")).status_code != 200 else "Found"
    }

def analyze_with_gpt(website_data):
    """Uses GPT-4o to analyze design, SEO, and performance issues."""
    prompt = f"""
    Analyze the following website content for SEO, UI/UX, and performance issues:

    URL: {website_data['url']}
    Title: {website_data['title']}
    Meta Description: {website_data['meta_desc']}
    H1 Tags: {website_data['h1_tags']}
    Images without ALT tags: {website_data['alt_images']}
    Canonical URL: {website_data['canonical_url']}
    Robots.txt: {website_data['robots_txt']}
    Sitemap.xml: {website_data['sitemap_xml']}
    
    Perform the following checks:
    - SEO weaknesses
    - UI/UX issues (font problems, layout, etc.)
    - Performance issues (slow page load, large images, etc.)
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content  # ✅ Return GPT's analysis result

def generate_pdf_report(issues, filename="website_audit_report.pdf"):
    """Generates a PDF report listing all detected issues."""
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "📌 Website Audit Report")
    c.drawString(100, 730, "---------------------------------")

    y = 710
    for issue in issues.split("\n"):
        if y < 50:
            c.showPage()
            y = 750
        c.drawString(100, y, issue)
        y -= 20

    c.save()
    print(f"✅ Report saved as {filename}")

def crawl_all_pages(base_url):
    """Crawl all pages of the website."""
    visited = set()  # Set to track visited URLs
    pages_data = []

    def crawl_page(url):
        if url in visited:
            return
        visited.add(url)
        
        print(f"Crawling: {url}")
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extracting basic SEO data from the page
            page_data = crawl_website(url)  # Reusing your original crawl_website function
            pages_data.append(page_data)
            
            # Find all internal links on the page
            for link in soup.find_all("a", href=True):
                next_url = link["href"]
                # Resolve relative links to absolute URLs
                absolute_url = urljoin(url, next_url)
                
                # Only crawl internal links (same domain)
                if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                    crawl_page(absolute_url)
                    
        except Exception as e:
            print(f"Error crawling {url}: {e}")
        
        time.sleep(1)  # Be respectful with pauses to not overload servers

    crawl_page(base_url)  # Start crawling from the base URL
    return pages_data

# Define AI Agent for Automation
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define tools AI can use
tools = [
    Tool(name="Website Crawler", func=crawl_website, description="Extracts website content"),
    Tool(name="AI Analyzer", func=analyze_with_gpt, description="Analyzes website issues with AI")
]

# Create an Autonomous AI Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Crawl the website (e.g., "https://www.nike.com/in/")
website_url = "https://www.panchalivastra.com/"
all_website_data = crawl_all_pages(website_url)

# Analyze SEO & Performance for all pages
all_issues = []
for page_data in all_website_data:
    issues = analyze_with_gpt(page_data)
    all_issues.append(issues)

# Generate a PDF Report for all pages
generate_pdf_report("\n".join(all_issues), filename="full_website_audit_report.pdf")
