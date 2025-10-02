# Source: https://p.vco.sh/?id=1759404098684416

from json import dump, load
from pathlib import Path


class MarkovGenerator:
    def __init__(self) -> None:
        self._weights: dict[str, dict[str, int]] = {}

    def load_model(self, fp: str) -> None:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} does not exist.")

        with path.open("r", encoding="utf-8") as f:
            self._weights = load(f)

    def save_model(self, fp: str) -> None:
        path = Path(fp)

        with path.open("w", encoding="utf-8") as f:
            dump(self._weights, f, ensure_ascii=False, indent=4, sort_keys=True)

    def _inc_weight(self, prev: str, token: str) -> None:
        if prev not in self._weights:
            self._weights[prev] = {}

        if token not in self._weights[prev]:
            self._weights[prev][token] = 0

        self._weights[prev][token] += 1

    def train_from_tokens(self, tokens: list[str]) -> None:
        prev = None
        for token in tokens:
            self._inc_weight(prev or "", token)
            prev = token

    def train_from_text(self, text: str, skip: str = ",.:;'\"![]()*&", case_insensitive: bool = True) -> None:
        token = ""
        tokens: list[str] = []

        if case_insensitive:
            text = text.lower()

        for char in text:
            if char.isspace():
                if token:
                    tokens.append(token)
                    token = ""
            elif char in skip:
                continue
            else:
                token += char

        if token:
            tokens.append(token)

        self.train_from_tokens(tokens)

    def generate(self, start: str, max_tokens: int = 100, seed: int | None = None) -> list[str]:
        import random

        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random

        current = start
        output = [current]

        for _ in range(max_tokens - 1):
            if current not in self._weights:
                break

            choices, weights = zip(*self._weights[current].items())
            current = rand.choices(choices, weights=weights)[0]
            output.append(current)

        return output
