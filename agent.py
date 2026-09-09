from dotenv import load_dotenv
load_dotenv()

import asyncio
import os

import aiohttp
from cogwit_sdk import cogwit, CogwitConfig
from openai import OpenAI

COGNEE_API_KEY = os.environ["COGNEE_API_KEY"]
COGWIT_API_BASE = os.getenv("COGWIT_API_BASE", "https://api.cognee.ai")

client = cogwit(CogwitConfig(api_key=COGNEE_API_KEY))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"

DATASET_NAME = "sales-support"


def _is_error(result) -> bool:
    return type(result).__name__.endswith("Error")


async def _add_via_api(text: str, dataset_name: str) -> dict:
    """Workaround for cogwit-sdk 0.1.7 bugs: its add() hits an unversioned
    path (/api/add instead of /api/v1/add) and sends the wrong field type
    (a plain string where the server requires an UploadFile) -- confirmed
    via the server's own OpenAPI schema and direct testing."""
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field(
            "data",
            text.encode("utf-8"),
            filename="document.txt",
            content_type="text/plain",
        )
        form.add_field("datasetName", dataset_name)
        async with session.post(
            f"{COGWIT_API_BASE}/api/v1/add",
            headers={"X-Api-Key": COGNEE_API_KEY},
            data=form,
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"add failed: {data}")
            return data


async def _search_via_api(query: str, dataset_name: str) -> str:
    """Workaround for cogwit-sdk 0.1.7 bugs: its search() hits an
    unversioned path and never sends a `datasets` field at all, so it can
    never actually scope a search to a specific dataset -- confirmed via
    the server's own OpenAPI schema and direct testing."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COGWIT_API_BASE}/api/v1/search",
            headers={"X-Api-Key": COGNEE_API_KEY, "Content-Type": "application/json"},
            json={
                "search_type": "CHUNKS",
                "query": query,
                "datasets": [dataset_name],
            },
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return ""
    parts = []
    for dataset_result in data or []:
        for item in dataset_result.get("search_result", []):
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _rewrite_query(user_message: str, conversation: list) -> str:
    """Use recent turns to make a standalone, fully-specified question --
    restores the effect of cross-turn continuity without needing it from
    the search backend itself."""
    if not conversation:
        return user_message

    recent = conversation[-6:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given the recent conversation and a new question, rewrite the "
                    "new question so it's fully self-contained and makes sense on "
                    "its own, with no pronouns or implicit references to earlier "
                    "messages. If it's already self-contained, return it unchanged. "
                    "Return ONLY the rewritten question, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"RECENT CONVERSATION:\n{history_text}\n\nNEW QUESTION:\n{user_message}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def run_agent(user_message: str, conversation: list = None) -> str:
    search_query = _rewrite_query(user_message, conversation or [])
    context = asyncio.run(_search_via_api(search_query, DATASET_NAME))

    if not context:
        return "I don't have that information in memory."

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the information given below. You may "
                    "reasonably summarize, explain, or connect ideas from it -- "
                    "the question doesn't need to use the exact same wording as "
                    "the source material. What you must NOT do is invent specific "
                    "facts, numbers, or details that aren't actually present below. "
                    "Only say you don't have the information if what's given is "
                    "genuinely unrelated to the question -- not merely phrased "
                    "differently. Answer naturally and conversationally, as if you "
                    "simply know this; never refer to 'the context' or mention that "
                    "you're reading from retrieved information.\n\n"
                    f"KNOWN INFORMATION:\n{context}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


async def _cognee_remember(text: str) -> None:
    await _add_via_api(text, DATASET_NAME)
    cognify_result = await client.cognify(datasets=[DATASET_NAME])
    if _is_error(cognify_result):
        raise RuntimeError(f"cognify failed: {cognify_result}")


def remember(text: str) -> None:
    asyncio.run(_cognee_remember(text))


async def _cognee_ingest(text: str, on_progress=None) -> None:
    if on_progress:
        on_progress(20, "Uploading document to memory...")
    await _add_via_api(text, DATASET_NAME)

    if on_progress:
        on_progress(55, "Building knowledge graph (this can take a minute)...")
    cognify_result = await client.cognify(datasets=[DATASET_NAME])
    if _is_error(cognify_result):
        raise RuntimeError(f"cognify failed: {cognify_result}")

    if on_progress:
        on_progress(100, "Done.")


def ingest_file(text: str, on_progress=None) -> None:
    asyncio.run(_cognee_ingest(text, on_progress))