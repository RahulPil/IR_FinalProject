from typing import Dict, List, Set

from openai import OpenAI
from .config import load_config, get_openai_api_key

_cfg = load_config()

client = OpenAI(api_key=get_openai_api_key())


def _call_openai_chat(messages, n: int = 1, temperature: float = 0.7) -> List[str]:
    model = _cfg["expansion"]["model"]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
    )

    outputs = []
    for choice in resp.choices:
        content = choice.message.content
        outputs.append(content.strip())
    return outputs


def _normalize_expansion(text: str) -> str:
    return text.strip().lower()


def _accumulate_unique_candidates(
    original_query: str,
    system_prompt: str,
    user_instruction: str,
    pool_size: int,
    global_seen: Set[str],
) -> List[str]:
    if pool_size <= 0:
        return []

    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {
        "role": "user",
        "content": f"Original query: {original_query}\n\n{user_instruction}",
    }

    expansions: List[str] = []
    local_seen: Set[str] = set()
    max_rounds = 5
    rounds = 0

    while len(expansions) < pool_size and rounds < max_rounds:
        needed = pool_size - len(expansions)
        raw_outputs = _call_openai_chat(
            messages=[system_msg, user_msg],
            n=needed,
            temperature=_cfg["expansion"].get("temperature", 0.7),
        )

        for out in raw_outputs:
            s = out.strip()
            if s.startswith('"') and s.endswith('"') and len(s) > 2:
                s = s[1:-1].strip()

            norm = _normalize_expansion(s)
            if not norm:
                continue
            if norm in global_seen or norm in local_seen:
                continue

            expansions.append(s)
            global_seen.add(norm)
            local_seen.add(norm)

            if len(expansions) >= pool_size:
                break

        rounds += 1

    return expansions


def _generate_paraphrase_pool(
    original_query: str,
    pool_size: int,
    global_seen: Set[str],
) -> List[str]:
    system_prompt = (
        "You rewrite search queries. Given a user query, generate alternative "
        "paraphrased versions using different wording but preserving meaning. "
        "Do NOT answer the question. Only rewrite it as a search-style query."
    )
    user_instruction = (
        "Generate one paraphrased search query that preserves intent but "
        "uses different phrasing."
    )
    return _accumulate_unique_candidates(
        original_query, system_prompt, user_instruction, pool_size, global_seen
    )


def _generate_entity_pool(
    original_query: str,
    pool_size: int,
    global_seen: Set[str],
) -> List[str]:
    system_prompt = (
        "You are a search query expansion assistant focused on entities and keywords. "
        "Given a user query, you identify important entities, technical terms, or "
        "keywords, and produce expanded queries that explicitly mention those terms "
        "and closely related entities or synonyms. Do NOT answer the question."
    )
    user_instruction = (
        "Generate one expanded search query that focuses on key entities/keywords, "
        "possibly including alternative names, acronyms, or directly related entities. "
        "Keep it a single search-style query."
    )
    return _accumulate_unique_candidates(
        original_query, system_prompt, user_instruction, pool_size, global_seen
    )


def _generate_conceptual_pool(
    original_query: str,
    pool_size: int,
    global_seen: Set[str],
) -> List[str]:
    system_prompt = (
        "You are a search query expansion assistant focusing on conceptual and "
        "broader-topic expansions. Given a user query, you generate related search "
        "queries that explore broader concepts, closely related ideas, or neighboring "
        "topics that a researcher might also want to retrieve. Do NOT answer the question."
    )
    user_instruction = (
        "Generate one conceptual or broader-topic search query that a user interested "
        "in this query might also search for. It should be clearly related, but it can "
        "be slightly more general or cover a neighboring concept. Keep it a single "
        "search-style query."
    )
    return _accumulate_unique_candidates(
        original_query, system_prompt, user_instruction, pool_size, global_seen
    )

def generate_expansions(
    original_query: str,
    num_paraphrase: int = None,
    num_entity: int = None,
    num_conceptual: int = None,
) -> Dict[str, List[str]]:
    exp_cfg = _cfg["expansion"]

    pool_paraphrase = exp_cfg.get("pool_size_paraphrase", 6)
    pool_entity = exp_cfg.get("pool_size_entity", 6)
    pool_conceptual = exp_cfg.get("pool_size_conceptual", 6)

    global_seen: Set[str] = set()

    paraphrases = _generate_paraphrase_pool(original_query, pool_paraphrase, global_seen)
    entities = _generate_entity_pool(original_query, pool_entity, global_seen)
    conceptuals = _generate_conceptual_pool(original_query, pool_conceptual, global_seen)

    expansions = {
        "paraphrase": paraphrases,
        "entity": entities,
        "conceptual": conceptuals,
    }

    return expansions
