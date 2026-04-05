import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import pandas as pd
from datetime import datetime

# =========================
# SETTINGS
# =========================
WHOOGLE_URL = "http://192.168.1.38:5000/search?q="
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"   # change if needed
MAX_URLS = 5                 # how many URLs to process
MAX_TEXT_LENGTH = 5000       # limit page text sent to LLM


# =========================
# STEP 1: SEARCH WHOOGLE
# =========================
def search_whoogle(query: str):
    """
    Search Whoogle and return a list of URLs from the result page.
    """
    url = WHOOGLE_URL + quote(query)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.startswith("http://") or href.startswith("https://"):
            if href not in links:
                links.append(href)

    return links[:MAX_URLS]


# =========================
# STEP 2: EXTRACT WEBPAGE TEXT
# =========================
def extract_webpage_text(url: str):
    """
    Download webpage and extract title + visible text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # clean spaces

        return {
            "source_link": url,
            "page_title": title,
            "page_text": text[:MAX_TEXT_LENGTH]
        }

    except Exception as e:
        print(f"Error reading {url}: {e}")
        return None


# =========================
# STEP 3: ASK OLLAMA TO EXTRACT FIELDS
# =========================
def extract_with_ollama(page_data: dict):
    """
    Send webpage text to Ollama and ask it to return JSON only.
    """
    prompt = f"""
Extract the following information from the webpage content below.

Return JSON only.
Do not add explanation.
If a field is missing, return No information.
If there are multiple values, return them as a list.

Required fields:
- title
- date
- company_name
- capability
- weapon_name
- aircraft_type
- location
- contact
- abstract
- source_link
- cost_or_prices
- technical_specifications

Webpage title:
{page_data["page_title"]}

Source link:
{page_data["source_link"]}

Webpage text:
{page_data["page_text"]}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an information extraction assistant. "
                    "Always return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        content = result["message"]["content"].strip()

        # try direct JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # try fixing if model returns ```json ... ```
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        # make sure source link exists
        data["source_link"] = page_data["source_link"]

        return data

    except Exception as e:
        print(f"Error with Ollama for {page_data['source_link']}: {e}")
        return None


# =========================
# STEP 4: SAVE RESULTS
# =========================
def save_results(results):
    """
    Save results to Excel and JSON.
    """
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"extracted_results_{timestamp}.xlsx"
    json_file = f"extracted_results_{timestamp}.json"

    df.to_excel(excel_file, index=False)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nSaved Excel: {excel_file}")
    print(f"Saved JSON : {json_file}")


# =========================
# MAIN
# =========================
def main():
    query = input("Enter your search query: ")

    print(f"\nSearching Whoogle for: {query}")
    urls = search_whoogle(query)

    print("\nCollected URLs:")
    for i, link in enumerate(urls, 1):
        print(f"{i}. {link}")

    all_results = []

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Reading: {url}")

        page_data = extract_webpage_text(url)
        if not page_data:
            continue

        extracted_data = extract_with_ollama(page_data)
        if extracted_data:
            all_results.append(extracted_data)
            print("Extracted successfully.")
        else:
            print("Failed to extract.")

    print("\nFinal extracted data:")
    print(json.dumps(all_results, indent=4, ensure_ascii=False))

    save_results(all_results)


if __name__ == "__main__":
    main()