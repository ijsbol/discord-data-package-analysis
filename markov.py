
import glob
import json
from pathlib import Path

from libmarkov import MarkovGenerator


gen = MarkovGenerator()


def train(gen: MarkovGenerator) -> None:
    channels: list[str] = glob.glob("Messages/c*")
    message_json_files: list[str] = [f"{channel}/messages.json" for channel in channels]
    messages_count: int = 0

    for message_file in message_json_files:
        if not Path(message_file).exists():
            continue

        print(f"Opening {message_file=} | {messages_count:,} {' '*10}", end='\r')
        with open(message_file, "r") as f:
            messages = json.load(f)
            messages_count += len(messages)
            for message in messages:
                if len(message["Contents"]) == 0:
                    continue
                gen.train_from_text(message["Contents"], skip="")

    gen.save_model("markov_model.json")


if Path("markov_model.json").exists():
    gen.load_model("markov_model.json")
else:
    train(gen)

# Non-seeded (non-deterministic) output
while True:
    output = gen.generate(input("Enter a prompt: (can be empty): "), max_tokens=1000)
    print(" ".join(output))
    print("\n\n")
