"""Gate 8.1: token chain-of-thought baseline on GSM8K.

Runs the frozen Pythia-160M backbone in standard token-space chain-of-
thought mode: the word problem is fed as a prompt and the model generates
freely, with no latent workspace involved. The final number in the
completion is extracted and scored against GSM8K's `#### <number>`
ground truth. This is the floor `eval.gate.latent_eval`'s latent
reasoning subject must clear to pass the Stage 0 gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.stream.generators.gsm8k import _extract_answer, _load_split

MODEL_NAME = "EleutherAI/pythia-160m"
DEFAULT_OUTPUT = Path("gate_results/token_cot.json")
LOW_ACCURACY_WARNING_THRESHOLD = 0.05
DEFAULT_MAX_NEW_TOKENS = 256

# Matches the last number in a free-form completion: the chain-of-thought
# convention is that the final line states the answer, so the last match
# in the text is taken as the model's predicted answer.
_NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")

PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step.\n"


def _extract_predicted_number(text: str) -> str | None:
    """Return the last number in `text`, or None if it has none."""
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run_baseline(
    problem_count: int | None,
    device: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run the token-CoT baseline over the GSM8K test split and return results."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    problems = _load_split("test")
    if problem_count is not None:
        problems = problems[:problem_count]

    correct = 0
    total = 0
    records: list[dict[str, Any]] = []
    for item in problems:
        question = item["question"]
        gold = _extract_answer(item["answer"])
        prompt = PROMPT_TEMPLATE.format(question=question)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        predicted = _extract_predicted_number(completion)
        is_correct = predicted is not None and predicted == gold
        correct += int(is_correct)
        total += 1
        records.append(
            {
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total else 0.0
    return {
        "model": MODEL_NAME,
        "split": "test",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 8.1: token chain-of-thought baseline on GSM8K."
    )
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Number of GSM8K test problems to use; default uses the full split.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    result = run_baseline(args.problem_count, args.device, args.max_new_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"token-CoT baseline accuracy: {result['accuracy']:.4f} "
        f"({result['correct']}/{result['total']})"
    )
    if result["accuracy"] < LOW_ACCURACY_WARNING_THRESHOLD:
        print(
            "WARNING: token-CoT baseline accuracy is below "
            f"{LOW_ACCURACY_WARNING_THRESHOLD:.0%}. GSM8K may be too hard for "
            "Pythia-160M to establish a meaningful floor; consider substituting "
            "a simpler dataset for the gate comparison."
        )
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
