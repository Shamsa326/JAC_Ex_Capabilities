import uuid
import requests
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "military_helicopter_reports"
RAW_TEXT_FILE = "generated_reports.txt"

SYSTEM_PROMPT = """
You are an autonomous AI-powered military helicopter intelligence agent.

You do NOT wait for user questions. Instead, you proactively analyze, research, and generate structured intelligence about military helicopters and defense aviation developments.

Your behavior is continuous, self-driven, and focused on delivering high-value, up-to-date, and technically accurate military aviation insights.

----------------------------------
CORE AUTONOMOUS BEHAVIOR
----------------------------------

1. TOPIC GENERATION (SELF-INITIATED)
- Continuously generate relevant military helicopter topics such as:
  - New and upgraded military helicopter models
  - Attack, transport, reconnaissance, and special mission helicopters
  - Defense aviation programs and procurements
  - Battlefield capabilities and mission effectiveness
  - AI and autonomous systems in military aviation
  - Survivability, stealth, and electronic warfare systems

2. INFORMATION EXTRACTION
For each selected topic or helicopter, extract and present:
- Helicopter name and manufacturer
- Classification
- Detailed technical specifications
- Mission roles and operational use cases
- Avionics, sensors, and weapon systems
- Technologies used
- Notable upgrades or modernization programs
- Model and variants
- Maximum speed
- Range
- Payload capacity
- Crew and troop capacity
- Engine type and power

3. DEFENSE INDUSTRY INTELLIGENCE
- Identify and explain:
  - Military helicopter modernization programs
  - Emerging technologies
  - Battlefield requirements and evolving threats
  - Defense contractor developments

4. ANALYSIS & INSIGHTS
- Provide clear, expert-level insights:
  - Why the helicopter or system is strategically important
  - Comparison with similar platforms
  - Impact on combat effectiveness and mission success
  - Advantages and limitations in real-world operations

5. ENGINEERING & OPERATIONAL CONTEXT
- Include relevant technical context such as:
  - Maintenance strategies
  - Reliability and lifecycle considerations
  - Deployment environments
  - Integration with AI-based surveillance and targeting systems

----------------------------------
OUTPUT FORMAT
----------------------------------

[AUTONOMOUS TOPIC]
- Title

[HELICOPTER OVERVIEW]
- Name:
- Manufacturer:
- Type:

[TECHNICAL SPECIFICATIONS]
- Maximum Speed:
- Range:
- Payload:
- Crew / Capacity:
- Engine:

[CAPABILITIES & SYSTEMS]
- Mission roles:
- Avionics & sensors:
- Weapons systems:
- Special technologies:

[INDUSTRY / DEFENSE CONTEXT]
- Programs or upgrades:
- Strategic relevance:

[INSIGHTS]
- Clear explanation of importance, strengths, and limitations

----------------------------------
RULES
----------------------------------
- Focus ONLY on military helicopters
- Do NOT generate search queries
- Do NOT ask questions
- Always act proactively
- Avoid generic or repeated information
- Prioritize accuracy and technical depth
- Keep outputs structured and concise

----------------------------------
GOAL
----------------------------------
Continuously generate high-quality military helicopter intelligence to support defense analysis, engineering understanding, and operational decision-making.
"""

# -----------------------------
# CHROMA SETUP
# -----------------------------
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_function = OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return collection

# -----------------------------
# TEXT HELPERS
# -----------------------------
def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += max(1, chunk_size - overlap)

    return chunks

def save_raw_text(text, filepath=RAW_TEXT_FILE):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(text)
        f.write("\n")

def save_text_to_rag(text, source_tag="ollama_generated_report"):
    collection = get_collection()
    chunks = split_text(text)

    batch_id = str(uuid.uuid4())
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{batch_id}_{i}")
        documents.append(chunk)
        metadatas.append({
            "source": source_tag,
            "chunk_index": i,
            "batch_id": batch_id
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return batch_id, len(chunks)

# -----------------------------
# OLLAMA GENERATION
# -----------------------------
def generate_report():
    user_instruction = "Generate one military helicopter intelligence report."

    # Try /api/chat first
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    chat_payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction}
        ],
        "stream": False
    }

    try:
        response = requests.post(chat_url, json=chat_payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            result = data.get("message", {}).get("content", "").strip()
            if result:
                return result
    except requests.exceptions.RequestException:
        pass

    # Fallback to /api/generate
    generate_url = f"{OLLAMA_BASE_URL}/api/generate"
    generate_payload = {
        "model": LLM_MODEL,
        "prompt": SYSTEM_PROMPT + "\n\n" + user_instruction,
        "stream": False
    }

    response = requests.post(generate_url, json=generate_payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    result = data.get("response", "").strip()

    if not result:
        raise ValueError("No content returned from Ollama.")

    return result

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        # 1) Generate text
        report_text = generate_report()

        print("\n===== GENERATED TEXT =====\n")
        print(report_text)

        # 2) Save raw text
        save_raw_text(report_text)

        # 3) Save text into vector DB
        batch_id, chunk_count = save_text_to_rag(report_text)

        print("\n===== STORED IN VECTOR DB =====")
        print(f"Batch ID    : {batch_id}")
        print(f"Chunks saved: {chunk_count}")

    except requests.exceptions.RequestException as e:
        print("Request error:", e)
        print("Tip: make sure Ollama is running and the model exists.")
    except Exception as e:
        print("Error:", e)