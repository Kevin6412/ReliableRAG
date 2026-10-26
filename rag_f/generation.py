"""Optional local, frozen language model for answer generation."""
from __future__ import annotations

INSUFFICIENT = 'INSUFFICIENT_INFORMATION'


def make_prompt(question: str, documents: list[str], max_doc_chars: int = 340) -> str:
    context = '\n'.join(f'[{i}] {doc[:max_doc_chars]}' for i, doc in enumerate(documents, 1))
    return ("Answer the question using ONLY the evidence passages below. "
            f"If the evidence does not answer the question, output exactly {INSUFFICIENT}. "
            "Give only a short answer, without explanation.\n"
            f"Question: {question}\nEvidence:\n{context}\nAnswer:")


class FrozenGenerator:
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache = {}

    def generate(self, question: str, docs: list[str]) -> str:
        if not docs:
            return INSUFFICIENT
        prompt = make_prompt(question, docs)
        if prompt in self.cache:
            return self.cache[prompt]
        tokens = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                                max_length=512).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**tokens, max_new_tokens=48, do_sample=False)
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        self.cache[prompt] = answer
        return answer


def is_abstention(text: str) -> bool:
    normalized = text.strip().casefold()
    return (not normalized or 'insufficient_information' in normalized or
            "i don't know" in normalized or "i do not know" in normalized or
            'insufficient information' in normalized or 'cannot determine' in normalized)


def answer_match(prediction, gold) -> bool:
    """Cheap lexical proxy only; not a validated faithfulness metric.

    RGB answers may be a string, list[str], or list[list[str]] (aliases).
    """
    if is_abstention(prediction):
        return False
    import re
    def canon(x):
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(x).casefold())).strip()
    text = f' {canon(prediction)} '
    if not text.strip():
        return False
    def matches_string(value):
        term = canon(value)
        return bool(term) and f' {term} ' in text
    # Match RGB author's checkanswer() semantics: all top-level answer
    # components must be present; a nested list offers alternative aliases.
    if not isinstance(gold, list):
        return matches_string(gold)
    return all(any(matches_string(alias) for alias in item)
               if isinstance(item, list) else matches_string(item)
               for item in gold)
