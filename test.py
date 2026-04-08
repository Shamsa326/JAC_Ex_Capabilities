import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# =========================
# SETTINGS
# # =========================
# WHOOGLE_URL = "http://localhost:5000/search?q="
WHOOGLE_URL = "http://172.28.32.1:5000/search?q="
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tinyllama"   # change if needed
   #"phi-3-mini"

def search_whoogle(query: str):
    """
    Search Whoogle and return a list of URLs from the result page.
    """
    url = WHOOGLE_URL + quote(query)
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



def main():
    query = input("Enter your search query: ")

    print(f"\nSearching Whoogle for: {query}")
    urls = search_whoogle(query)

    print("\nCollected URLs:")
    for i, link in enumerate(urls, 1):
        print(f"{i}. {link}")



if __name__ == "__main__":
    main()