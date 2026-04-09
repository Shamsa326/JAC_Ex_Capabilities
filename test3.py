import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import pandas as pd
from datetime import datetime
import os

# =========================
# SETTINGS
# =========================
WHOOGLE_URL = "http://172.28.32.1:5000/search?q="
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5"      # change if needed
MAX_URLS_PER_QUERY = 5        # how many URLs from each generated query
MAX_TOTAL_URLS = 10           # total unique URLs to process
MAX_TEXT_LENGTH = 5000        # limit page text sent to LLM
NUM_SEARCH_QUERIES = 5        # how many optimized queries LLM should generate

# Memory file to store previous user prompts
MEMORY_FILE = "memory_prompts.json"


# =========================
# MEMORY FUNCTIONS
# =========================
def load_memory():
    """
    Load previous user prompts from memory file.
    """
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        return []

    except Exception as e:
        print(f"Error loading memory: {e}")
        return []


def save_prompt_to_memory(user_prompt: str):
    """
    Save current user prompt into memory file if not already saved.
    """
    memory = load_memory()

    # avoid duplicate exact prompt
    if user_prompt not in memory:
        memory.append(user_prompt)

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving memory: {e}")


def build_memory_context():
    """
    Build memory knowledge text from saved prompts.
    """
    memory = load_memory()

    if not memory:
        return "No prior memory knowledge."

    # take latest prompts only if memory becomes too long
    recent_memory = memory[-10:]

    memory_text = "\n".join([f"- {item}" for item in recent_memory])
    return memory_text


# =========================
# STEP 1: GENERATE SEARCH QUERIES WITH OLLAMA
# =========================
def generate_search_queries(user_prompt: str, memory_knowledge: str):
    """
    Use Ollama to generate optimized web search queries
    based on user prompt + prior memory knowledge.
    Returns a list of search queries.
    """
    prompt = f"""
You are a search query generation assistant.

Your task:
Generate optimized, specific web search queries.

Input 1: User prompt
Input 2: Prior memory knowledge

Use both inputs to understand the topic and generate high-quality search queries.

Rules:
- Return exactly {NUM_SEARCH_QUERIES} search queries
- Make queries short, specific, and search-engine friendly
- Avoid duplicates
- Include technical keywords where useful
- Do not explain anything
- Return JSON only
- Output format must be:

{{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}}

User prompt:
{user_prompt}

Prior memory knowledge:
{memory_knowledge}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate optimized web search queries. "
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

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        queries = data.get("queries", [])

        clean_queries = []
        for q in queries:
            if isinstance(q, str):
                q = q.strip()
                if q and q not in clean_queries:
                    clean_queries.append(q)

        if not clean_queries:
            return [user_prompt]

        return clean_queries

    except Exception as e:
        print(f"Error generating search queries with Ollama: {e}")
        return [user_prompt]


# =========================
# STEP 2: SEARCH WHOOGLE
# =========================
def search_whoogle(query: str):
    """
    Search Whoogle and return a list of URLs from the result page.
    """
    try:
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

        return links[:MAX_URLS_PER_QUERY]

    except Exception as e:
        print(f"Error searching Whoogle for query '{query}': {e}")
        return []


# =========================
# STEP 3: SEARCH ALL GENERATED QUERIES
# =========================
def collect_urls_from_generated_queries(search_queries):
    """
    Search Whoogle using all generated queries
    and return unique URLs.
    """
    all_urls = []

    for i, query in enumerate(search_queries, 1):
        print(f"\nSearching Whoogle with generated query {i}/{len(search_queries)}:")
        print(f"  {query}")

        urls = search_whoogle(query)

        for url in urls:
            if url not in all_urls:
                all_urls.append(url)

        if len(all_urls) >= MAX_TOTAL_URLS:
            break

    return all_urls[:MAX_TOTAL_URLS]


# =========================
# STEP 4: EXTRACT WEBPAGE TEXT
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
# STEP 5: ASK OLLAMA TO EXTRACT FIELDS
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

Required fields and keep them in same order:
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

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

        data["source_link"] = page_data["source_link"]

        return data

    except Exception as e:
        print(f"Error with Ollama for {page_data['source_link']}: {e}")
        return None


# =========================
# STEP 6: SAVE RESULTS
# =========================
def save_results(results):
    """
    Save results to CSV only.
    """
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"extracted_results_{timestamp}.csv"

    df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"\nSaved CSV: {csv_file}")


# =========================
# MAIN
# =========================
def main():
    # user enters only one prompt
    user_prompt = input("Enter user prompt: ").strip()

    if not user_prompt:
        print("No prompt entered.")
        return

    # load previous prompts as memory knowledge
    memory_knowledge = build_memory_context()

    print("\nLoaded memory knowledge:")
    print(memory_knowledge)

    # generate optimized search queries
    print("\nGenerating optimized search queries with Ollama...")
    generated_queries = generate_search_queries(user_prompt, memory_knowledge)

    print("\nGenerated search queries:")
    for i, q in enumerate(generated_queries, 1):
        print(f"{i}. {q}")

    # save current prompt to memory
    save_prompt_to_memory(user_prompt)

    # search Whoogle using generated queries
    print("\nSearching Whoogle using generated queries...")
    urls = collect_urls_from_generated_queries(generated_queries)

    print("\nCollected URLs:")
    for i, link in enumerate(urls, 1):
        print(f"{i}. {link}")

    # extract from webpages
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