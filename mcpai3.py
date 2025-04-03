import os
import requests
import time
import argparse
from collections import Counter
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import openai
import re
import subprocess  # To run your script as a subprocess
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.llms import OpenAI  # Or another LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Your existing code (with potential minor adjustments for output)
def extract_ui_performance_data(url):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        start_time = time.time()
        driver.get(url)
        load_time = time.time() - start_time

        headings = driver.find_elements(By.TAG_NAME, "h1")
        h1_texts = [h.text for h in headings if h.text]

        buttons = driver.find_elements(By.TAG_NAME, "button")
        button_texts = [b.text for b in buttons if b.text]

        links = driver.find_elements(By.TAG_NAME, "a")
        visible_links = [a.text for a in links if a.text]
        all_links = [a.get_attribute("href") for a in links if a.get_attribute("href")]

        driver.set_window_size(375, 812)
        mobile_view = "✅ Mobile Responsive" if driver.execute_script("return document.documentElement.scrollWidth <= window.innerWidth;") else "❌ Not Mobile Responsive"

        images = driver.find_elements(By.TAG_NAME, "img")
        missing_alt = [img.get_attribute("src") for img in images if not img.get_attribute("alt")]

        return {
            "h1_tags": h1_texts,
            "buttons": button_texts,
            "visible_links": visible_links,
            "all_links": all_links,
            "mobile_responsive": mobile_view,
            "page_load_time": round(load_time, 2),
            "missing_alt_images": missing_alt,
        }
    finally:
        driver.quit()

def crawl_website(url, max_pages=10):
    visited = set()
    to_visit = [url]
    all_data = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = " ".join([p.text for p in soup.find_all("p")])
            images = [img["src"] for img in soup.find_all("img") if img.get("src")]
            ui_data = extract_ui_performance_data(current_url)
            title = soup.title.text if soup.title else "No Title"
            meta_desc = soup.find("meta", attrs={"name": "description"})
            meta_desc = meta_desc["content"] if meta_desc else "No Meta Description"
            headers = {f"H{h.name[-1]}": [h.text.strip() for h in soup.find_all(h.name)] for h in soup.find_all(["h1", "h2", "h3"])}

            all_data.append({"url": current_url, "text": page_text, "images": images, "title": title, "meta_desc": meta_desc, "headers": headers, **ui_data})
            visited.add(current_url)

            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                if urlparse(full_url).netloc == urlparse(url).netloc and full_url not in visited:
                    to_visit.append(full_url)
        except:
            continue

    return all_data

def check_broken_links(links):
    broken_links = []
    for link in links:
        try:
            response = requests.head(link, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                broken_links.append(link)
        except:
            broken_links.append(link)
    return broken_links

def keyword_density(text):
    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)
    return word_counts.most_common(10)

def analyze_with_gpt(website_data):
    keyword_data = keyword_density(website_data["text"])
    broken_links = check_broken_links(website_data["all_links"])

    prompt = f"""
    Analyze the following website for SEO, UI/UX, and performance issues:

    **URL:** {website_data['url']}
    **Title:** {website_data['title']}
    **Meta Description:** {website_data['meta_desc']}
    **H1 Tags:** {website_data['h1_tags']}
    **Page Load Time:** {website_data['page_load_time']} seconds
    **Mobile Responsive:** {website_data['mobile_responsive']}
    **Broken Links:** {broken_links}
    **Top Keywords:** {keyword_data}

    🎯 **Analysis Focus:**
    - Optimize title & meta description.
    - Identify broken links & missing ALT attributes.
    - Detect keyword stuffing.
    - Suggest UI improvements for accessibility.
    - Recommend performance optimizations.
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def generate_pdf_report(issues, filename="website_audit_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    content = [Paragraph("📌 Website Audit Report", styles["Title"]), Spacer(1, 12)]
    for section in issues.split("\n\n"):
        content.append(Paragraph(section, styles["Normal"]))
        content.append(Spacer(1, 10))
    doc.build(content)
    print(f"✅ Report saved as {filename}")
    return filename # Return the filename for the agent

# Agent Tool Definition
def run_website_audit(website_url):
    """Runs the website audit script on the given URL and generates a PDF report."""
    try:
        # Run the script, capturing output
        result = subprocess.run(
            ["python", __file__, website_url],  # Assuming your script is in the same file
            capture_output=True,
            text=True,
            check=True
        )
        # Extract the report filename from the output
        output_lines = result.stdout.strip().split('\n')
        report_location_line = [line for line in output_lines if "Report saved as" in line]
        if report_location_line:
            report_filename = report_location_line[0].split("as ")[1]
            return f"Website audit completed. Report saved at: {report_filename}"
        else:
            return "Website audit completed, but the report filename could not be extracted from the output."
    except subprocess.CalledProcessError as e:
        return f"Error running website audit: {e.stderr}"

# Initialize Langchain components
llm = OpenAI(temperature=0)  # You can choose a different LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [
    Tool(
        name="run_website_audit",
        func=run_website_audit,
        description="Runs a website audit on the given URL and generates a PDF report. Input should be the website URL.",
    )
]

prefix = """You are a helpful assistant that can perform website audits. Use the available tools to fulfill user requests. If the user asks to perform an audit, use the 'run_website_audit' tool. Once the audit is complete and the report is generated, inform the user of the report location."""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

# Example of how to use the Langchain AI Agent
if __name__ == "__main__":
    # You can still run your script directly for command-line usage
    parser = argparse.ArgumentParser(description="Website SEO & Performance Auditor")
    parser.add_argument("url", help="Website URL to analyze")
    args = parser.parse_args()

    website_data = crawl_website(args.url)
    issues = analyze_with_gpt(website_data[0])
    generate_pdf_report(issues, filename="website_audit_report1.pdf")

    # Or interact with the Langchain AI Agent
    agent_input = input("What would you like to do? ")
    agent_response = agent_chain.run(input=agent_input)
    print(agent_response)

    while True:
        agent_input = input("Anything else? ")
        if agent_input.lower() in ["exit", "quit", "done"]:
            break
        agent_response = agent_chain.run(input=agent_input)
        print(agent_response)