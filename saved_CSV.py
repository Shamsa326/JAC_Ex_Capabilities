import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import pandas as pd
from datetime import datetime

# =========================
# SETTINGS
# =========================
WHOOGLE_URL = "http://172.28.32.1:5000/search?q="
OLLAMA_URL = "http://localhost:11434/api/chat"

GEN_MODEL = "llama3"        # for query generation
EXTRACT_MODEL = "qwen2.5"   # for extraction

MAX_URLS = 5
MAX_TEXT_LENGTH = 10000

MEMORY_FILE = "helicopter_intel.txt"
URL_MEMORY_FILE = "processed_urls.txt"


# =========================
# LOAD MEMORY (PAST SEARCH)
# =========================
try:
    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        past_search = file.read()
except Exception:
    past_search = ""

print(f"\n📁 Past Search:\n{past_search}\n")


# =========================
# STEP 1: GENERATE QUERY (OLLAMA AGENT)
# =========================
def generate_query():
    system_prompt = f"""
You are an autonomous AI-powered military helicopter intelligence agent.

generate intelligence queries about military helicopters and defense aviation developments.

make query more related to helicopter detailes such as manufacturer,Classification (attack, utility, transport, SAR, special operations, etc.),
technical specifications,operational use cases ,weapon systems,Technologies used (AI, autonomy, targeting systems, defensive aids),Maximum speed,
Payload capacity,Range, Crew and troop capacity.

### make  only search queries in 15 words.

----------------------------------
RULES
----------------------------------
- Focus ONLY on military helicopters
- Do NOT ask questions
- Avoid generic or repeated information

### make  only search queries in 15 words.

 give me only 1 search query
 give me only 1 search query


----------------------------------
Past search
----------------------------------
You have made the following past search , DO Not repeat them 

Past search:
{past_search}
"""

    payload = {
        "model": GEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Start"}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        query = data.get("message", {}).get("content", "").strip()

        print("\n🧠 Generated Query:\n", query)

        # save query to memory
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(query + "\n")

        return query

    except Exception as e:
        print("Query generation error:", e)
        return None


# =========================
# URL MEMORY HELPERS
# =========================
def load_old_urls():
    if not os.path.exists(URL_MEMORY_FILE):
        return set()

    with open(URL_MEMORY_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_new_urls(urls):
    if not urls:
        return

    with open(URL_MEMORY_FILE, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


# =========================
# STEP 2: SEARCH WHOOGLE
# =========================
def search_whoogle(query: str):
    search_url = WHOOGLE_URL + quote(query)

    response = requests.get(search_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.startswith("http://") or href.startswith("https://"):
            if href not in links:
                links.append(href)

    old_urls = load_old_urls()

    # remove already processed URLs
    filtered_links = [link for link in links if link not in old_urls]
    duplicate_count = len(links) - len(filtered_links)

    # apply limit after filtering
    new_links = filtered_links[:MAX_URLS]

    print(f"\n🧠 Filtered {duplicate_count} duplicate URLs")

    return new_links


# =========================
# STEP 3: EXTRACT WEBPAGE TEXT
# =========================
def extract_webpage_text(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        return {
            "source_link": url,
            "page_title": title,
            "page_text": text[:MAX_TEXT_LENGTH]
        }

    except Exception as e:
        print(f"Error reading {url}: {e}")
        return None


# =========================
# STEP 4: EXTRACT WITH OLLAMA
# =========================
def extract_with_ollama(page_data: dict):
    prompt = f"""
Extract structured information from the webpage content below.

Return JSON only.
Do not add explanation.
If a field is missing, return "No information".
If there are multiple values, return them as a list.

Fields:
- title
- date
- goal
- abstract
- weapon_name
- aircraft_type
- capability
- company_name
- contact
- location
- cost
- source_link
- technical_specifications
- rest_information

Title: {page_data["page_title"]}
Link: {page_data["source_link"]}

Text:
{page_data["page_text"]}
"""

    payload = {
        "model": EXTRACT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are an information extraction assistant. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        content = result["message"]["content"].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        # make sure source_link is always included
        data["source_link"] = page_data["source_link"]

        return data

    except Exception as e:
        print(f"Extraction error for {page_data['source_link']}: {e}")
        return None


# =========================
# STEP 5: SAVE CSV
# =========================
def save_results(results):
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)

    folder_name = "Extracted_files"
    os.makedirs(folder_name, exist_ok=True)

    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(folder_name, filename)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 Saved: {file_path}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    # 1. Generate query automatically
    query = generate_query()
    if not query:
        return

    # 2. Search
    print("\n🔎 Searching Whoogle...")
    urls = search_whoogle(query)

    if not urls:
        print("No new URLs found.")
        return

    print("\n🌐 New URLs:")
    for i, u in enumerate(urls, 1):
        print(f"{i}. {u}")

    results = []

    # 3. Process URLs
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")

        page = extract_webpage_text(url)
        if not page:
            print("❌ Failed to read webpage")
            continue

        data = extract_with_ollama(page)
        if data:
            results.append(data)

            # save URL only after successful extraction
            save_new_urls([url])

            print("✅ Extracted")
        else:
            print("❌ Failed to extract")

    # 4. Show final data
    print("\n📊 Final Data:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 5. Save CSV
    save_results(results)


if __name__ == "__main__":
    main()