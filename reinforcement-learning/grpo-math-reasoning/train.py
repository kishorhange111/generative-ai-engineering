"""
Teaching an LLM to reason with reinforcement learning: GRPO on GSM8K (DeepSeek-R1 style)
=======================================================================================

QUESTION: Can we improve a model's math reasoning WITHOUT showing it any
worked solutions - only telling it whether its final answer was right?

GRPO - Group Relative Policy Optimization (DeepSeek-Math 2024, used for DeepSeek-R1):
    For each question the model samples a GROUP of G answers (here 8).
    Each answer gets a reward from simple, verifiable rules:
        +1.0  final answer equals the ground truth           (correctness)
        +0.5  output follows <think>...</think><answer>...</answer> (format)
    The advantage of each answer = (its reward - group mean) / group std.
    Answers better than their siblings are reinforced, worse ones suppressed.
    Compared with PPO (the original RLHF algorithm) there is NO value/critic
    network - the group itself is the baseline - so it needs about half the
    memory and is far simpler. A KL penalty to the starting model (beta)
    keeps the policy from drifting too far.

    "Reinforcement learning with verifiable rewards" (RLVR): no human labels,
    no reward model - correctness is checked by code. This is how current
    reasoning models (o1/R1 style) are trained, at a much larger scale.

SETUP: Qwen2.5-1.5B-Instruct, GSM8K grade-school math (7.5k train / 1.3k test),
Hugging Face TRL's GRPOTrainer, bf16 on one A100, fixed time budget.
EVALUATION: greedy decoding on held-out GSM8K test questions, same prompt
before and after RL: answer accuracy and format compliance.

The reward functions and answer parsers live in `src/genai_common/reasoning.py` and are
unit-tested in `tests/test_reasoning.py` - in RL, a bug in the reward is a bug
in what the model learns.
"""

import json
import os
import sys
import time

import torch

# Make <repo>/src importable so `genai_common` works without `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # <repo>/src
from genai_common.reasoning import (
    correctness_reward,
    extract_answer,
    format_reward,
    is_well_formatted,
    same_number,
    to_prompt,
)

QUICK = os.environ.get("QUICK") == "1"  # 4-step smoke test
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUT = os.environ.get("OUT", "/content/grpo_gsm8k")
N_TEST = 64 if QUICK else int(os.environ.get("N_TEST", "500"))
MAX_STEPS = 4 if QUICK else int(os.environ.get("MAX_STEPS", "400"))
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.2"))  # whichever comes first: steps or time


@torch.no_grad()  # inference only
def evaluate(model, tok, test, label):
    """Greedy-decode every test question and report accuracy and format compliance.

    Returns (metrics, samples) where samples are 6 example outputs for the README.
    """
    model.eval()
    tok.padding_side = "left"  # left padding for batched generation with a decoder-only model
    correct = formatted = 0
    samples, t = [], time.time()
    for i in range(0, len(test), 32):
        batch = test.select(range(i, min(i + 32, len(test))))
        texts = [tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False) for ex in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # 512 new tokens leaves room for a full chain of thought.
        out = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
        for ex, g in zip(batch, out[:, enc["input_ids"].shape[1] :]):  # keep only the generated part
            text = tok.decode(g, skip_special_tokens=True)
            ok = same_number(extract_answer(text), ex["answer"])
            correct += ok
            formatted += is_well_formatted(text)
            if len(samples) < 6:
                samples.append(
                    {
                        "question": ex["prompt"][1]["content"],
                        "gold": ex["answer"],
                        "output": text[-700:],
                        "correct": bool(ok),
                    }
                )
    res = {
        "accuracy": correct / len(test),
        "format_compliance": formatted / len(test),
        "eval_minutes": (time.time() - t) / 60,
    }
    print(label, res, flush=True)
    return res, samples


def main():
    """Evaluate -> GRPO training (time-budgeted) -> evaluate -> save metrics, samples and logs."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    # Only the question and the final number are kept: the worked solutions are never used for training.
    train = ds["train"].shuffle(seed=42).map(to_prompt, remove_columns=ds["train"].column_names)
    test = ds["test"].shuffle(seed=42).select(range(N_TEST)).map(to_prompt, remove_columns=ds["test"].column_names)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # sdpa = PyTorch's fused scaled-dot-product attention (FlashAttention-style kernels).
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    base_res, base_samples = evaluate(model, tok, test, "before RL")
    tok.padding_side = "left"  # GRPO also generates, so keep left padding

    class TimeBudget(TrainerCallback):
        """Stop training once TIME_BUDGET_HOURS of wall-clock time are used."""

        def __init__(self):
            self.deadline = time.time() + TIME_BUDGET_HOURS * 3600

        def on_step_end(self, args, state, control, **k):
            """Request a stop after the current step once the deadline has passed."""
            if time.time() > self.deadline:
                print(f"[time budget] stop at step {state.global_step}")
                control.should_training_stop = True
            return control

    args = GRPOConfig(
        # RL fine-tuning uses a tiny LR: large policy updates quickly collapse generation quality.
        output_dir=OUT,
        max_steps=MAX_STEPS,
        learning_rate=2e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.03,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=8,  # G: answers sampled per question (16 x 2 / 8 = 4 questions per step)
        # temperature 0.9: enough diversity inside a group for rewards (and advantages) to differ.
        max_completion_length=384,
        temperature=0.9,
        beta=0.04,  # KL penalty to the starting model
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    # Rewards are summed: max 1.5 per completion (correct + well formatted).
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[correctness_reward, format_reward],
        args=args,
        train_dataset=train,
        callbacks=[TimeBudget()],
    )
    t = time.time()
    trainer.train()
    train_min = (time.time() - t) / 60

    tuned_res, tuned_samples = evaluate(trainer.model, tok, test, "after GRPO")
    summary = {
        "model": MODEL,
        "steps": trainer.state.global_step,
        "questions_seen": trainer.state.global_step * 4,
        "group_size": 8,
        "train_minutes": train_min,
        "gpu": torch.cuda.get_device_name(0),
        "test_questions": len(test),
        "before": base_res,
        "after": tuned_res,
    }
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"before": base_samples, "after": tuned_samples}, open(os.path.join(OUT, "samples.json"), "w"), indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
