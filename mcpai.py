import os
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import openai
from reportlab.pdfgen import canvas

# Load API key securely from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def crawl_website(url):
    """Extracts text, images, and meta info from the website."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    page_text = " ".join([p.text for p in soup.find_all("p")])
    images = [img["src"] for img in soup.find_all("img") if img.get("src")]

    return {
        "text": page_text,
        "images": images,
        "title": soup.title.text if soup.title else "No Title",
        "meta_desc": soup.find("meta", attrs={"name": "description"}) or "No Meta Description"
    }

def analyze_with_gpt(website_data):
    """Uses GPT-4o to analyze design, SEO, and performance issues."""
    prompt = f"""
    Analyze the following website content for SEO, UI/UX, and performance issues:
    
    Title: {website_data['title']}
    Meta Description: {website_data['meta_desc']}
    Text Content: {website_data['text'][:1000]}  # Limiting for efficiency
    Images: {website_data['images'][:5]} (First 5 images)
    
    Check for:
    - Bad fonts
    - Poor image sizing
    - Readability issues
    - Mobile responsiveness problems
    - SEO weaknesses
    - UX flaws
    """
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": prompt}]
)

    return response.choices[0].message.content  # ✅ Fix applied


    return response["choices"][0]["message"]["content"]

def generate_pdf_report(issues, filename="wwebsite_audit_report.pdf"):
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

# Run the AI Agent
website_url = "https://www.adidas.co.in/"
website_data = crawl_website(website_url)
issues = analyze_with_gpt(website_data)

# Generate Report
generate_pdf_report(issues)
