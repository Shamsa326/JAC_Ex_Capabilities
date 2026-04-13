
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

File_NAME = "helicopter_intel.txt"
try:
    with open(File_NAME, "r", encoding="utf-8") as file:
        content = file.read()
except Exception as e:
        content = ""

print(f"past search: {content}")

SYSTEM_PROMPT = f"""You are an autonomous AI-powered military helicopter intelligence agent.

generate intelligence queries about military helicopters and defense aviation developments.



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

{content}

"""

def run_agent():
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Start"}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        data = response.json()

        text_output = data.get("message", {}).get("content", "")

        print("\n===== GENERATED TEXT =====\n")
        print(text_output)

        # ✅ Save as text file
        with open(File_NAME, "a", encoding="utf-8") as f:
            f.write(text_output)

        print("\n✅ Saved to helicopter_intel.txt")

        return text_output

    except requests.exceptions.RequestException as e:
        print("Request error:", e)
        return None


if __name__ == "__main__":
    run_agent()
    