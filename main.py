import requests
from bs4 import BeautifulSoup

# =========================
# SETTINGS
# =========================
WHOOGLE_URL = "http://localhost:5000/search?q="
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tinyllama"   # change if needed
   #"phi-3-mini"

def search_whoogle(query: str):
    """
    Search Whoogle and return a list of URLs from the result page.
    """
    url = WHOOGLE_URL + query
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    # Try to find result links
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Keep only http/https links
        if href.startswith("http://") or href.startswith("https://"):
            if href not in links:
                links.append(href)

    return links[:10]


def ask_ollama(prompt: str):
    """
    Send a test prompt to Ollama chat API.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


def main():
    query = "aircraft enhancement companies"

    print(f"\nSearching Whoogle for: {query}")
    urls = search_whoogle(query)

    print("\nCollected URLs:")
    for i, link in enumerate(urls, 1):
        print(f"{i}. {link}")

    print("\nTesting Ollama...")
    answer = ask_ollama("Say: Ollama is working successfully.")
    print("\nOllama response:")
    print(answer)


if __name__ == "__main__":
    main()