"""Prompts for grounded answering.

Prompt design here is a correctness concern, not a stylistic one. The system's
entire claim is that answers trace to specific abstracts, and the prompt is what
enforces that at generation time.

Three properties the prompt must produce:

1. **Refusal.** The model must say the context is insufficient rather than fall
   back on parametric knowledge. This is the hard part: a model that knows a
   plausible answer will supply it, and the answer will be unattributable and
   often subtly wrong for the specific compound asked about.
2. **Inline citation.** Claims carry [PMID] markers so a reader can check any
   individual sentence, not just the reference list at the bottom.
3. **No inter-source synthesis that the sources don't support.** Two abstracts
   reporting different IC50 values for the same compound is a real disagreement
   to surface, not an average to compute.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are a biomedical literature assistant answering questions about kinase \
inhibitors from retrieved PubMed abstracts.

ABSOLUTE RULES:

1. Answer ONLY from the provided abstracts. You have extensive pretrained \
knowledge of this field; do not use it. If the abstracts do not contain the \
answer, say so explicitly. An unattributable answer is a failure, even when it \
is correct.

2. Cite inline with [PMID] after every factual claim, e.g. \
"MK-2206 inhibits AKT1 with an IC50 of 12 nM [30000001]." Every number, \
compound name, and mechanism needs its citation on the sentence that states it.

3. If abstracts disagree, report the disagreement and cite both. Do not average \
values, pick a favourite, or silently drop the inconvenient one.

4. Do not extrapolate. If an abstract reports activity against PI3K-alpha, do \
not infer anything about PI3K-beta. Related-protein inference is exactly the \
error this system exists to prevent.

5. Be concise. Two to four sentences unless the question demands more.

If the context is insufficient, respond with exactly:
INSUFFICIENT_CONTEXT: <one sentence naming what is missing>
"""

ANSWER_TEMPLATE = """\
Retrieved abstracts:

{context}

---

Question: {question}

Answer using only the abstracts above, citing [PMID] inline."""

# Used by evaluation/groundedness.py — a second, cheap LLM pass that checks
# whether an answer is actually supported by its cited context. Deliberately
# framed as verification of a claim in isolation, because asking the same model
# that wrote the answer "was that right?" mostly measures its agreeableness.
GROUNDEDNESS_SYSTEM = """\
You verify whether a claim is supported by a source text. Answer with exactly \
one word: SUPPORTED, CONTRADICTED, or NOT_MENTIONED.

SUPPORTED means the source states the claim or directly entails it.
CONTRADICTED means the source states something incompatible.
NOT_MENTIONED means the source is silent on it.

Do not use outside knowledge. Judge only against the source text given.\
"""

GROUNDEDNESS_TEMPLATE = """\
Source text:
{source}

Claim:
{claim}

One word: SUPPORTED, CONTRADICTED, or NOT_MENTIONED."""


def format_context(chunks) -> str:
    """Render retrieved chunks for the prompt.

    PMID is placed at the START of each block, not buried in a trailing
    citation. Models copy identifiers far more reliably when the identifier is
    adjacent to the text it labels — burying it after 200 words of abstract
    measurably increases mis-citation.
    """
    blocks = []
    for hit in chunks:
        chunk = hit.chunk if hasattr(hit, "chunk") else hit
        header = f"[PMID {chunk.pmid}] {chunk.title}"
        meta = f"({chunk.journal}, {chunk.pub_year})" if chunk.journal else ""
        blocks.append(f"{header} {meta}\n{chunk.text}")
    return "\n\n".join(blocks)
