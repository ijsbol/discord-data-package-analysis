# Made with ❤️ by vcokltfre

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
            dump(self._weights, f, ensure_ascii=False, indent=4)

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

    def train_from_text_2layer(self, text: str, skip: str = ",.:;'\"![]()*&", case_insensitive: bool = True) -> None:
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

        prev1 = None
        prev2 = None
        for token in tokens:
            key = (prev1 or "", prev2 or "")
            self._inc_weight(f"{key[0]} {key[1]}", token)
            prev1, prev2 = prev2, token

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

    def generate_2layer(self, start: str, max_tokens: int = 100, seed: int | None = None) -> list[str]:
        import random

        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random

        parts = start.split()
        if len(parts) >= 2:
            prev1, prev2 = parts[-2], parts[-1]
        elif len(parts) == 1:
            prev1, prev2 = "", parts[-1]
        else:
            prev1, prev2 = "", ""

        output = [prev1, prev2] if prev1 else [prev2]

        for _ in range(max_tokens - 2):
            key = f"{prev1} {prev2}"
            if key not in self._weights:
                break

            choices, weights = zip(*self._weights[key].items())
            current = rand.choices(choices, weights=weights)[0]
            output.append(current)
            prev1, prev2 = prev2, current

        return output
