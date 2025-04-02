import os
import requests
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from reportlab.pdfgen import canvas
import openai
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔹 UI & Performance Analysis
def extract_ui_performance_data(url):
    """Extracts UI, performance, and accessibility details using Selenium."""
    options = Options()
    options.add_argument("--headless")  # Run headless for speed
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # Start timer to measure page load time
        start_time = time.time()
        driver.get(url)
        load_time = time.time() - start_time  # Calculate page load time

        # Extract UI elements
        headings = driver.find_elements(By.TAG_NAME, "h1")
        h1_texts = [h.text for h in headings if h.text]

        buttons = driver.find_elements(By.TAG_NAME, "button")
        button_texts = [b.text for b in buttons if b.text]

        links = driver.find_elements(By.TAG_NAME, "a")
        visible_links = [a.text for a in links if a.text]

        # Extract font sizes (CSS computed styles)
        font_sizes = {}
        for h in headings:
            font_size = driver.execute_script("return window.getComputedStyle(arguments[0]).fontSize;", h)
            font_sizes[h.text] = font_size
        
        # Check mobile responsiveness
        driver.set_window_size(375, 812)  # Mobile viewport
        mobile_view = "✅ Mobile Responsive" if driver.execute_script("return document.documentElement.scrollWidth <= window.innerWidth;") else "❌ Not Mobile Responsive"

        # Check accessibility - Missing ALT text
        images = driver.find_elements(By.TAG_NAME, "img")
        missing_alt = [img.get_attribute("src") for img in images if not img.get_attribute("alt")]

        # Measure total image size
        total_image_size = sum(
            int(requests.head(img.get_attribute("src")).headers.get("content-length", 0)) for img in images if img.get_attribute("src")
        ) / (1024 * 1024)  # Convert to MB

        return {
            "h1_tags": h1_texts,
            "buttons": button_texts,
            "visible_links": visible_links,
            "font_sizes": font_sizes,
            "mobile_responsive": mobile_view,
            "page_load_time": round(load_time, 2),
            "missing_alt_images": missing_alt,
            "total_image_size_mb": round(total_image_size, 2)
        }

    finally:
        driver.quit()

# 🔹 Website Crawler
def crawl_website(url):
    """Extracts text, images, meta info, and UI data from the website."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    page_text = " ".join([p.text for p in soup.find_all("p")])
    images = [img["src"] for img in soup.find_all("img") if img.get("src")]

    # Extract UI & Performance Data using Selenium
    ui_data = extract_ui_performance_data(url)

    return {
        "url": url,
        "text": page_text,
        "images": images,
        "title": soup.title.text if soup.title else "No Title",
        "meta_desc": soup.find("meta", attrs={"name": "description"})["content"] if soup.find("meta", attrs={"name": "description"}) else "No Meta Description",
        **ui_data
    }

# 🔹 AI-Powered Analysis (SEO, UI/UX, Performance)
def analyze_with_gpt(website_data):
    """Uses GPT-4o to analyze design, SEO, and performance issues."""
    prompt = f"""
    Analyze the following website for SEO, UI/UX, and performance issues:

    URL: {website_data['url']}
    Title: {website_data['title']}
    Meta Description: {website_data['meta_desc']}
    H1 Tags: {website_data['h1_tags']}
    Buttons: {website_data['buttons']}
    Visible Links: {website_data['visible_links']}
    Font Sizes: {website_data['font_sizes']}
    Mobile Responsive: {website_data['mobile_responsive']}
    Page Load Time: {website_data['page_load_time']} seconds
    Missing ALT Images: {website_data['missing_alt_images']}
    Total Image Size: {website_data['total_image_size_mb']} MB
    
    **Perform the following checks:**
    - SEO issues (missing meta tags, poor keyword optimization)
    - UI/UX problems (bad layout, font readability, broken buttons)
    - Performance issues (slow load times, large images)
    - Accessibility issues (low contrast, missing ALT text, poor keyboard navigation)
    
    ✅ Provide SPECIFIC, actionable recommendations based on the data above.
    ✅ DO NOT just say "use PageSpeed Insights"—directly analyze and give solutions.
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content  # ✅ Returns GPT's analysis result

# 🔹 Generate PDF Report
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

# 🔹 Define AI Agent for Automation
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define tools AI can use
tools = [
    Tool(name="Website Crawler", func=crawl_website, description="Extracts website content, UI, and performance details."),
    Tool(name="AI Analyzer", func=analyze_with_gpt, description="Analyzes website issues with AI.")
]

# Create an AI Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔹 Start Crawling & Analyzing
website_url = "https://www.panchalivastra.com/"
all_website_data = [crawl_website(website_url)]

# Analyze SEO & Performance for all pages
all_issues = []
for page_data in all_website_data:
    issues = analyze_with_gpt(page_data)
    all_issues.append(issues)

# Generate PDF Report
generate_pdf_report("\n".join(all_issues), filename="full_website_audit_report.pdf")
