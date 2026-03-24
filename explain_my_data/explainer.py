"""
Groq API integration for Explain My Data.
Uses llama-3.3-70b-versatile for text and llama-4-scout for chart vision.
"""

from groq import Groq

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are a concise data analyst. Respond in bullet points only — no prose paragraphs.

Rules:
- Every response is bullets only. Group under short bold headers if needed.
- One fact per bullet, max 15 words each.
- Always cite actual numbers and column names.
- Lead with the most important insight.
- No filler phrases ("It's worth noting...", "In conclusion...", "Overall...").
"""


def _get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def explain_dataset(data_summary: str, api_key: str) -> str:
    """Generate a full plain-English explanation of the uploaded dataset."""
    client = _get_client(api_key)

    user_message = f"""Dataset summary:

{data_summary}

Give me ≤8 bullets covering: data shape, key numeric patterns, categorical distributions, data quality issues (missing/duplicates/outliers), one surprising finding, and one recommended next step.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content


def explain_chart(chart_title: str, data_summary: str, api_key: str) -> str:
    """Generate a plain-English explanation of a chart given its title and dataset context."""
    client = _get_client(api_key)

    user_message = f"""Chart: '{chart_title}'

Dataset summary:
{data_summary}

Give ≤4 bullets: what the chart shows, the key trend or anomaly (with numbers), and one plain-English takeaway.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def answer_question(question: str, data_summary: str, api_key: str) -> str:
    """Answer a free-form user question about their dataset."""
    client = _get_client(api_key)

    user_message = f"""Dataset summary:
{data_summary}

Question: {question}

Answer directly in ≤3 bullets. Cite specific numbers or column names.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content
