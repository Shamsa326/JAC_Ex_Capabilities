import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse, parse_qs, unquote, urljoin
import json
import pandas as pd
from datetime import datetime
import time
from multiprocessing import Process, Manager

# =========================
# SETTINGS
# =========================
WHOOGLE_URL = "http://172.28.32.1:5000/search?q="
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
OLLAMA_URL = "http://localhost:11434/api/chat"

GEN_MODEL = "llama3"          # for query generation
EXTRACT_MODEL = "qwen2.5" # for extraction
COMPARE_MODEL = "qwen2.5"     # for comparing old CSV data vs new extracted data

MAX_URLS = 10
MAX_TEXT_LENGTH = 12000
MAX_COMPARE_OLD = 20
MAX_COMPARE_NEW = 10
SLEEP_SECONDS = 300   #5 minutes


# Shared global blocklist file for all agents
GLOBAL_URL_MEMORY_FILE = "global_processed_urls.txt"


# =========================
# FILE / MEMORY HELPERS
# =========================
def load_past_search(memory_file):
    try:
        with open(memory_file, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return ""


def load_urls_from_file(file_path):
    if not os.path.exists(file_path):
        return set()

    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_url_to_file(file_path, url):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def reserve_global_url(url, shared_url_dict, global_lock):
    with global_lock:
        if url in shared_url_dict:
            return False

        shared_url_dict[url] = True
        append_url_to_file(GLOBAL_URL_MEMORY_FILE, url)
        return True


def load_old_urls(url_memory_file):
    return load_urls_from_file(url_memory_file)


def save_new_urls(urls, url_memory_file):
    if not urls:
        return

    with open(url_memory_file, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


# =========================
# QUERY GENERATION
# =========================
def clean_generated_query(raw_query: str):
    if not raw_query:
        return None

    lines = [line.strip() for line in raw_query.splitlines() if line.strip()]
    if not lines:
        return None

    query = lines[0].strip().strip('"').strip("'")

    bad_prefixes = [
        "here is",
        "here's",
        "this is",
        "search query:",
        "query:"
    ]

    lower_q = query.lower()
    for prefix in bad_prefixes:
        if lower_q.startswith(prefix):
            parts = query.split(":", 1)
            if len(parts) == 2:
                query = parts[1].strip().strip('"').strip("'")
            break

    return query if query else None


def generate_query(system_prompt, memory_file):
    past_search = load_past_search(memory_file)

    full_system_prompt = f"""
{system_prompt}

----------------------------------
Past search
----------------------------------
You have made the following past search. Do NOT repeat them.

Past search:
{past_search}

IMPORTANT:
- Return ONLY the search query
- Do NOT add explanation
- Do NOT add quotes around the whole answer
- Do NOT write: "Here is the query" or similar
- Output one line only
"""

    payload = {
        "model": GEN_MODEL,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": "Generate one clean search query only."}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()

        data = response.json()
        raw_query = data.get("message", {}).get("content", "").strip()
        query = clean_generated_query(raw_query)

        print(f"\n🧠 Generated Query [{memory_file}]:\n{query}")

        if query:
            with open(memory_file, "a", encoding="utf-8") as f:
                f.write(query + "\n")

        return query

    except Exception as e:
        print(f"Query generation error [{memory_file}]: {e}")
        return None


# =========================
# URL FILTERING
# =========================
def is_bad_url(url: str) -> bool:
    if not url:
        return True

    u = url.strip().lower()

    blocked_keywords = [
        "google.com/maps",
        "maps.google",
        "/maps/",
        "google.com/search",
        "/search?",
        "webcache",
        "googleusercontent",
        "javascript:",
        "mailto:",
        "github.com/benbusby/whoogle-search",
        "google.com/",
        "www.google.com/",
        "tbm=isch",
        "tbm=vid",
        "tbm=nws",
        "/preferences",
        "/opensearch",
        "/favicon",
    ]

    for bad in blocked_keywords:
        if bad in u:
            return True

    return False


def looks_like_real_page(url: str) -> bool:
    if not url:
        return False

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if not parsed.netloc:
        return False

    bad_domains = {
        "google.com",
        "www.google.com",
        "github.com",
        "www.github.com",
        "maps.google.com"
    }

    if parsed.netloc.lower() in bad_domains:
        return False

    return True


def normalize_url(url: str):
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        query = parsed.query
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return url


def debug_first_links(soup, limit=30):
    print("\n===== RAW SEARCH HREFS =====")
    anchors = soup.find_all("a", href=True)

    if not anchors:
        print("No <a href> tags found.")
        return

    for i, a in enumerate(anchors[:limit], 1):
        print(f"{i}. {a.get('href', '')}")


# =========================
# WHOOGLE PARSER
# =========================
def clean_result_url_whoogle(href: str, whoogle_base: str):
    if not href:
        return None

    href = href.strip()

    if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
        return None

    if href.startswith("http://") or href.startswith("https://"):
        url = unquote(href)
        if is_bad_url(url):
            return None
        if looks_like_real_page(url):
            return normalize_url(url)
        return None

    if href.startswith("/"):
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)

        for key in ["q", "url", "u", "target"]:
            real_url = qs.get(key, [None])[0]
            if real_url:
                real_url = unquote(real_url)
                if real_url.startswith(("http://", "https://")):
                    if not is_bad_url(real_url) and looks_like_real_page(real_url):
                        return normalize_url(real_url)

        absolute_url = urljoin(whoogle_base, href)

        if is_bad_url(absolute_url):
            return None

        if urlparse(absolute_url).netloc == urlparse(whoogle_base).netloc:
            return None

        if looks_like_real_page(absolute_url):
            return normalize_url(absolute_url)

    if href.startswith("//"):
        absolute_url = "https:" + href
        if not is_bad_url(absolute_url) and looks_like_real_page(absolute_url):
            return normalize_url(absolute_url)

    return None


def whoogle_navigation_only(anchors):
    raw_hrefs = [a.get("href", "").strip() for a in anchors]

    nav_patterns = [
        "home?preferences=",
        "tbm=isch",
        "tbm=vid",
        "tbm=nws",
        "maps.google.com/maps?q=",
        "//www.google.com/",
        "github.com/benbusby/whoogle-search",
    ]

    matched = sum(
        1 for href in raw_hrefs
        if any(p in href for p in nav_patterns)
    )

    return len(anchors) <= 7 and matched >= 5


def search_whoogle_only(query: str):
    search_url = WHOOGLE_URL + quote(query)
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers, timeout=30)
    response.raise_for_status()

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    print(f"\n🔎 Query: {query}")
    print(f"🌐 Whoogle request URL: {search_url}")
    print(f"📄 HTML length: {len(html)}")

    anchors = soup.find_all("a", href=True)
    print(f"🔗 Total anchors found in page: {len(anchors)}")

    if whoogle_navigation_only(anchors):
        print("⚠️ Whoogle returned navigation-only page, not real search results.")
        debug_first_links(soup)
        return [], True

    links = []
    seen = set()
    whoogle_base = WHOOGLE_URL.split("/search?q=")[0]

    for a in anchors:
        href = a.get("href", "").strip()
        real_url = clean_result_url_whoogle(href, whoogle_base)

        if real_url and real_url not in seen:
            seen.add(real_url)
            links.append(real_url)

    links = [link for link in links if not is_bad_url(link) and looks_like_real_page(link)]

    print(f"🧾 Whoogle clean links: {len(links)}")
    for i, link in enumerate(links[:10], 1):
        print(f"   {i}. {link}")

    if not links:
        debug_first_links(soup)

    return links, False


# =========================
# DUCKDUCKGO HTML FALLBACK
# =========================
def search_duckduckgo_only(query: str):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(
        DUCKDUCKGO_HTML_URL,
        headers=headers,
        data={"q": query},
        timeout=30
    )
    response.raise_for_status()

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    print(f"\n🦆 DuckDuckGo fallback for query: {query}")
    print(f"📄 HTML length: {len(html)}")

    links = []
    seen = set()

    anchors = soup.find_all("a", href=True)
    print(f"🔗 Total anchors found in DuckDuckGo page: {len(anchors)}")

    for a in anchors:
        href = a.get("href", "").strip()

        classes = a.get("class", [])
        text = a.get_text(" ", strip=True).lower()

        candidate = None

        if href.startswith("http://") or href.startswith("https://"):
            candidate = href

        elif "uddg=" in href:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            real_url = qs.get("uddg", [None])[0]
            if real_url:
                candidate = unquote(real_url)

        if candidate:
            candidate = normalize_url(candidate)
            if candidate not in seen and not is_bad_url(candidate) and looks_like_real_page(candidate):
                seen.add(candidate)
                links.append(candidate)

    print(f"🧾 DuckDuckGo clean links: {len(links)}")
    for i, link in enumerate(links[:10], 1):
        print(f"   {i}. {link}")

    return links


# =========================
# SEARCH WRAPPER
# =========================
def search_web(query: str, url_memory_file: str, shared_url_dict, global_lock):
    links = []
    used_source = "Whoogle"

    try:
        whoogle_links, blocked = search_whoogle_only(query)
        if blocked or not whoogle_links:
            print("⚠️ Switching to DuckDuckGo fallback...")
            links = search_duckduckgo_only(query)
            used_source = "DuckDuckGo"
        else:
            links = whoogle_links
    except Exception as e:
        print(f"⚠️ Whoogle error: {e}")
        print("⚠️ Switching to DuckDuckGo fallback...")
        try:
            links = search_duckduckgo_only(query)
            used_source = "DuckDuckGo"
        except Exception as e2:
            print(f"Fallback search error: {e2}")
            return []

    print(f"✅ Search source used: {used_source}")

    agent_old_urls = load_old_urls(url_memory_file)
    filtered_links = [link for link in links if link not in agent_old_urls]
    print(f"🧠 After agent memory filtering: {len(filtered_links)}")

    with global_lock:
        global_old_urls = set(shared_url_dict.keys())

    filtered_links = [link for link in filtered_links if link not in global_old_urls]
    print(f"🧠 Global blocklist size: {len(global_old_urls)}")
    print(f"🧠 After global filtering: {len(filtered_links)}")

    new_links = filtered_links[:MAX_URLS]
    print(f"✅ Final new links: {len(new_links)}")

    return new_links


# =========================
# WEBPAGE EXTRACTION
# =========================
def extract_webpage_text(url: str):
    try:
        if "github.com" in url.lower():
            print(f"⏭️ Skipping GitHub page: {url}")
            return None

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=40)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        if not text.strip():
            print(f"⚠️ Empty page text for: {url}")
            return None

        return {
            "source_link": url,
            "page_title": title,
            "page_text": text[:MAX_TEXT_LENGTH]
        }

    except Exception as e:
        print(f"Error reading {url}: {e}")
        return None


# =========================
# OLLAMA EXTRACTION
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        content = result.get("message", {}).get("content", "").strip()

        if not content:
            print(f"Extraction error: empty response for {page_data['source_link']}")
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        data["source_link"] = page_data["source_link"]
        return data

    except Exception as e:
        print(f"Extraction error for {page_data['source_link']}: {e}")
        return None


# =========================
# CSV HELPERS
# =========================
def load_old_csv_data(output_folder):
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
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

    print(f"\n📚 Loaded {len(all_old_records)} old records from CSV files in {output_folder}")
    return all_old_records


def save_results(results, output_folder, agent_name):
    if not results:
        print(f"No results for {agent_name}.")
        return None

    df = pd.DataFrame(results)

    os.makedirs(output_folder, exist_ok=True)

    filename = f"{agent_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(output_folder, filename)

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 Saved CSV: {file_path}")
    return file_path


def filter_truly_new_records(old_records, new_records):
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
# COMPARE OLD VS NEW
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
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
# SAVE TEXT OUTPUT
# =========================
def save_text_output(text_output, new_folder, agent_name):
    os.makedirs(new_folder, exist_ok=True)

    filename = f"{agent_name}_whats_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(new_folder, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_output)

    print(f"\n📝 Saved TEXT file: {file_path}")
    return file_path


def append_summary_to_memory(text_output, memory_file):
    try:
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write(f"WHAT IS NEW SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(text_output + "\n")

        print(f"\n🧠 Appended summary to {memory_file}")
    except Exception as e:
        print(f"Could not append summary to {memory_file}: {e}")


# =========================
# MAIN PIPELINE
# =========================
def main(agent_name, system_prompt, shared_url_dict, global_lock):
    memory_file = f"{agent_name}_helicopter_intel.txt"
    url_memory_file = f"{agent_name}_processed_urls.txt"
    output_folder = f"{agent_name}_Extracted_files"
    new_folder = f"{agent_name}_New_Information"

    old_csv_records = load_old_csv_data(output_folder)

    query = generate_query(system_prompt, memory_file)
    if not query:
        print(f"[{agent_name}] No clean query generated.")
        return

    print(f"\n🔎 [{agent_name}] Searching web...")
    urls = search_web(query, url_memory_file, shared_url_dict, global_lock)

    if not urls:
        print(f"[{agent_name}] No new URLs found.")
        return

    print(f"\n🌐 [{agent_name}] New URLs:")
    for i, u in enumerate(urls, 1):
        print(f"{i}. {u}")

    results = []

    for i, url in enumerate(urls, 1):
        print(f"\n[{agent_name}] [{i}/{len(urls)}] Processing: {url}")

        with global_lock:
            if url in shared_url_dict:
                print(f"⏭️ [{agent_name}] Already globally processed: {url}")
                continue

        page = extract_webpage_text(url)
        if not page:
            print("❌ Failed to read webpage")
            continue

        data = extract_with_ollama(page)
        if data:
            reserved = reserve_global_url(url, shared_url_dict, global_lock)
            if not reserved:
                print(f"⏭️ [{agent_name}] Lost race, already reserved globally: {url}")
                continue

            results.append(data)
            save_new_urls([url], url_memory_file)
            print("✅ Extracted")
        else:
            print("❌ Failed to extract")

    print(f"\n📊 [{agent_name}] Final Data:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    csv_path = save_results(results, output_folder, agent_name)

    truly_new_results = filter_truly_new_records(old_csv_records, results)
    print(f"\n🆕 [{agent_name}] Locally detected new records: {len(truly_new_results)}")

    whats_new_text = generate_whats_new_text(old_csv_records, truly_new_results)

    print(f"\n========== [{agent_name}] WHAT IS NEW ==========")
    print(whats_new_text)

    txt_path = save_text_output(whats_new_text, new_folder, agent_name)
    append_summary_to_memory(whats_new_text, memory_file)

    print(f"\n✅ [{agent_name}] Done")
    print("CSV file :", csv_path)
    print("TEXT file:", txt_path)
    print("Memory   :", memory_file)


# =========================
# LOOP FOR EACH AGENT
# =========================
def run_agent_loop(agent_name, system_prompt, shared_url_dict, global_lock):
    while True:
        try:
            print(f"\n🚀 Starting new cycle for {agent_name}...\n")
            main(agent_name, system_prompt, shared_url_dict, global_lock)

            print(f"\n⏳ [{agent_name}] Waiting before next run...\n")
            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print(f"\n🛑 [{agent_name}] Stopped by user.")
            break

        except Exception as e:
            print(f"\n⚠️ Error in {agent_name} loop: {e}")
            print(f"[{agent_name}] Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    if not os.path.exists(GLOBAL_URL_MEMORY_FILE):
        with open(GLOBAL_URL_MEMORY_FILE, "w", encoding="utf-8") as f:
            pass

    manager = Manager()
    shared_url_dict = manager.dict()
    global_lock = manager.Lock()

    existing_global_urls = load_urls_from_file(GLOBAL_URL_MEMORY_FILE)
    for url in existing_global_urls:
        shared_url_dict[url] = True

    prompt_1 = """
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
"""

    prompt_2 = """
You are an autonomous AI-powered military helicopter intelligence agent.

Generate intelligence search queries about military helicopters. Make query more related to helicopter such as
AH-64D
AH-64E
AS350
AS365
AS550
AS565SB
AS565UB
AT504
AT802
AW109
AW139
B407GX
BL8K
C208B
C208B-EX
C208B-EX-VIP
C208-SP
CH-47C+
CH-47F
DH64SD
DHC6-300
DHC6-400
ECUREUIL
H145
KODI
SA330SM
SPUMA
T206H
UH-60L
UH-60M

### Make only search queries in 15 words.

----------------------------------
RULES
----------------------------------
- Focus ONLY on military helicopters
- Do NOT ask questions
- Avoid generic or repeated information
- Return only 1 search query
"""

    prompt_3 = """
You are an autonomous AI-powered military helicopter weapons system intelligence agent.

Generate intelligence search queries about helicopter weapon systems, including air-to-ground missiles,
rockets, gun systems, targeting pods, fire control systems, defensive aids, radars,
electronic warfare suites, countermeasures, and integrated weapon technologies for combat helicopters.

### Make only search queries in 15 words.

----------------------------------
RULES
----------------------------------
- Focus ONLY on military helicopters
- Do NOT ask questions
- Avoid generic or repeated information
- Return only 1 search query
"""

    p1 = Process(target=run_agent_loop, args=("agent1", prompt_1, shared_url_dict, global_lock))
    p2 = Process(target=run_agent_loop, args=("agent2", prompt_2, shared_url_dict, global_lock))
    p3 = Process(target=run_agent_loop, args=("agent3", prompt_3, shared_url_dict, global_lock))

    p1.start()
    p2.start()
    p3.start()

    try:
        p1.join()
        p2.join()
        p3.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all agents...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p1.join()
        p2.join()
        p3.join()
        print("✅ All agents stopped.")