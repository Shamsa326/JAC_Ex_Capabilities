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
COMPARE_MODEL = "qwen2.5"   # for comparing old CSV data vs new extracted data

MAX_URLS = 5
MAX_TEXT_LENGTH = 5000
MAX_COMPARE_OLD = 20
MAX_COMPARE_NEW = 10

MEMORY_FILE = "helicopter_intel.txt"
URL_MEMORY_FILE = "processed_urls.txt"
OUTPUT_FOLDER = "Extracted_files"


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

Generate intelligence search queries about military helicopters and defense aviation developments.

Make query more related to helicopter details such as manufacturer, classification
(attack, utility, transport, SAR, special operations, etc.),
technical specifications, operational use cases, weapon systems,
technologies used (AI, autonomy, targeting systems, defensive aids),
maximum speed, payload capacity, range, crew and troop capacity.

### Make only search queries in 15 words.

----------------------------------
RULES
----------------------------------
- Focus ONLY on military helicopters
- Do NOT ask questions
- Avoid generic or repeated information
- Return only 1 search query

----------------------------------
Past search
----------------------------------
You have made the following past search. Do NOT repeat them.

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

        # Save query to memory
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

    # Remove already processed URLs
    filtered_links = [link for link in links if link not in old_urls]
    duplicate_count = len(links) - len(filtered_links)

    # Apply limit after filtering
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
            {
                "role": "system",
                "content": "You are an information extraction assistant. Always return valid JSON only."
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

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        # Make sure source_link is always included
        data["source_link"] = page_data["source_link"]

        return data

    except Exception as e:
        print(f"Extraction error for {page_data['source_link']}: {e}")
        return None


# =========================
# STEP 5: LOAD OLD CSV DATA
# =========================
def load_old_csv_data():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    csv_files = [
        os.path.join(OUTPUT_FOLDER, f)
        for f in os.listdir(OUTPUT_FOLDER)
        if f.lower().endswith(".csv")
    ]

    all_old_records = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            records = df.fillna("").to_dict(orient="records")
            all_old_records.extend(records)
        except Exception as e:
            print(f"Could not read old CSV {file_path}: {e}")

    print(f"\n📚 Loaded {len(all_old_records)} old records from CSV files")
    return all_old_records


# =========================
# STEP 6: SAVE NEW CSV
# =========================
def save_results(results):
    if not results:
        print("No results.")
        return None

    df = pd.DataFrame(results)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 Saved CSV: {file_path}")
    return file_path


# =========================
# HELPER: SIMPLE LOCAL FILTER
# =========================
def filter_truly_new_records(old_records, new_records):
    """
    Local Python check before sending to Ollama.
    Removes exact duplicates based on source_link or title.
    """
    old_links = set()
    old_titles = set()

    for item in old_records:
        link = str(item.get("source_link", "")).strip().lower()
        title = str(item.get("title", "")).strip().lower()

        if link:
            old_links.add(link)
        if title:
            old_titles.add(title)

    unique_new = []
    for item in new_records:
        link = str(item.get("source_link", "")).strip().lower()
        title = str(item.get("title", "")).strip().lower()

        if link and link in old_links:
            continue
        if title and title in old_titles:
            continue

        unique_new.append(item)

    return unique_new


# =========================
# STEP 7: COMPARE OLD VS NEW WITH OLLAMA
# =========================
def generate_whats_new_text(old_records, new_records):
    if not new_records:
        return "No new extracted records were found."

    old_records_small = old_records[:MAX_COMPARE_OLD]
    new_records_small = new_records[:MAX_COMPARE_NEW]

    prompt = f"""
You are comparing old military helicopter intelligence records with new records.

Your task:
1. Find what is NEW in the new records compared with old records.
2. Focus on genuinely new information such as:
   - new helicopter model
   - new capability
   - new weapon system
   - new company/manufacturer information
   - new technical specification
   - new operational use
   - new technology
3. Ignore small wording differences.
4. Write the result as plain text only.
5. Keep the text clear and structured.

Return format:

WHAT IS NEW
- item 1
- item 2
- item 3

NEW SOURCES
- source 1
- source 2

If nothing important is new, write:
No significant new information found.

OLD RECORDS:
{json.dumps(old_records_small, ensure_ascii=False, indent=2)}

NEW RECORDS:
{json.dumps(new_records_small, ensure_ascii=False, indent=2)}
"""

    payload = {
        "model": COMPARE_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a comparison assistant. Return plain text only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()

        result = response.json()
        text_output = result.get("message", {}).get("content", "").strip()

        if not text_output:
            text_output = "No significant new information found."

        return text_output

    except Exception as e:
        print("Comparison error:", e)
        return "Comparison failed. Could not generate 'what is new' summary."


# =========================
# STEP 8: SAVE TEXT OUTPUT
# =========================
def save_text_output(text_output):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filename = f"whats_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_output)

    print(f"\n📝 Saved TEXT file: {file_path}")
    return file_path


# =========================
# STEP 9: APPEND TEXT OUTPUT TO MEMORY FILE
# =========================
def append_summary_to_memory(text_output):
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write(f"WHAT IS NEW SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(text_output + "\n")

        print(f"\n🧠 Appended summary to {MEMORY_FILE}")
    except Exception as e:
        print(f"Could not append summary to {MEMORY_FILE}: {e}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load old CSV records BEFORE saving new results
    old_csv_records = load_old_csv_data()

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

            # Save URL only after successful extraction
            save_new_urls([url])

            print("✅ Extracted")
        else:
            print("❌ Failed to extract")

    # 4. Show final data
    print("\n📊 Final Data:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 5. Save CSV
    csv_path = save_results(results)

    # 6. Filter exact duplicates locally first
    truly_new_results = filter_truly_new_records(old_csv_records, results)

    print(f"\n🆕 Locally detected new records: {len(truly_new_results)}")

    # 7. Generate "what is new" text with Ollama
    whats_new_text = generate_whats_new_text(old_csv_records, truly_new_results)

    print("\n========== WHAT IS NEW ==========")
    print(whats_new_text)

    # 8. Save text output in separate TXT file
    txt_path = save_text_output(whats_new_text)

    # 9. Append generated summary to helicopter_intel.txt
    append_summary_to_memory(whats_new_text)

    print("\n✅ Done")
    print("CSV file :", csv_path)
    print("TEXT file:", txt_path)
    print("Memory   :", MEMORY_FILE)


if __name__ == "__main__":
    main()