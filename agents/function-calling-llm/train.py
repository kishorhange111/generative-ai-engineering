"""
Fine-tuning a small LLM for function calling (the core skill of AI agents)
==========================================================================

TASK: given a set of available tools (JSON schemas) and a user request,
decide whether a tool is needed and, if so, emit the exact call:
    <functioncall>{"name": "get_exchange_rate", "arguments": {"base": "USD", "target": "INR"}}</functioncall>
or answer in plain text when no tool applies. This is what agent frameworks
(LangChain / LangGraph / CrewAI tools, OpenAI function calling, MCP tool use)
rely on - a wrong tool name or a malformed argument breaks the whole agent.

MODEL: Qwen2.5-1.5B-Instruct - small enough to run cheaply and locally,
which is exactly where fine-tuning pays off for agents.
DATA: glaive-function-calling-v2 (~113k conversations). We take each
conversation's first user turn and the assistant's first response (a function
call, or a plain answer when no function fits).
METHOD: LoRA (r=16) on all attention + MLP projections, bf16, loss on the
assistant response only.

METRICS on held-out conversations (same prompt before/after):
    call decision accuracy  - called a tool exactly when it should (and not otherwise)
    tool name accuracy      - right function, among cases that need one
    full-call exact match   - right function AND arguments equal as JSON
    valid JSON rate         - parsable call when a call was attempted

The data parser, call parser and metrics live in `common/function_calling.py`
and are unit-tested in `tests/test_function_calling.py`.
"""

import json
import os
import sys
import time

import torch

# Make the repo root importable so `common/` works from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.function_calling import call_metrics
from common.function_calling import fc_messages as messages  # messages(ex, with_answer)
from common.function_calling import parse_glaive_example as parse
from common.lm_data import masked_labels, pad_batch

QUICK = os.environ.get("QUICK") == "1"  # tiny smoke test
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUT = os.environ.get("OUT", "/content/function_calling")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "15000"))
N_TEST = 60 if QUICK else int(os.environ.get("N_TEST", "600"))
MAX_LEN = 1024  # tool schemas make prompts long
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "0.8"))


@torch.no_grad()  # inference only
def evaluate(model, tok, test, label):
    """Greedy-generate a reply for every test conversation and compute the agent metrics.

    Returns (metrics, samples) - see `common.function_calling.call_metrics` for the metric definitions.
    """
    model.eval()
    tok.padding_side = "left"  # batched generation with a decoder-only model needs left padding
    outputs, samples, t = [], [], time.time()
    for i in range(0, len(test), 16):
        batch = test.select(range(i, min(i + 16, len(test))))
        texts = [
            tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False) for ex in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for ex, g in zip(batch, out[:, enc["input_ids"].shape[1] :]):  # keep only the generated tokens
            text = tok.decode(g, skip_special_tokens=True)
            outputs.append(text)
            if len(samples) < 10:
                samples.append({"user": ex["user"][:300], "gold": ex["target"][:300], "pred": text[:300]})
    # ex["call"] is the gold call as JSON, or "" when the right answer is plain text.
    res = {**call_metrics(outputs, list(test["call"])), "eval_minutes": (time.time() - t) / 60}
    print(label, res, flush=True)
    return res, samples


def main():
    """Parse data -> zero-shot evaluation -> LoRA training -> evaluation -> save metrics and samples."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=42)
    # Parse a bit more than needed: some rows are dropped by the parser.
    raw = raw.select(range(min(len(raw), N_TRAIN + 3 * N_TEST + 2000)))
    data = raw.map(parse, remove_columns=raw.column_names).filter(lambda e: e["ok"])
    # Disjoint splits: the first N_TEST parsed rows are held out for testing.
    test = data.select(range(N_TEST))
    train = data.select(range(N_TEST, min(len(data), N_TEST + N_TRAIN)))
    print(f"train {len(train)}, test {len(test)} ({sum(bool(c) for c in test['call'])} need a call)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    base_res, base_samples = evaluate(model, tok, test, "zero-shot")

    # LoRA adapters on every attention and MLP projection; the base weights stay frozen.
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    # Needed with gradient checkpointing + frozen embeddings: otherwise no
    # tensor entering the checkpointed blocks requires grad and LoRA gets no gradient.
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()

    def encode(ex):
        """Tokenize prompt + target; only the assistant's reply contributes to the loss."""
        prompt = tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(messages(ex, True), tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        return {"input_ids": f_ids, "labels": masked_labels(p_ids, f_ids)}

    # Drop sequences whose answer was cut off entirely by MAX_LEN (all labels -100 -> nothing to learn).
    train_tok = train.map(encode, remove_columns=train.column_names).filter(
        lambda e: any(lab != -100 for lab in e["labels"])
    )

    def collate(b):
        """Pad to the longest sequence in the batch and convert to tensors."""
        return {k: torch.tensor(v) for k, v in pad_batch(b, tok.pad_token_id).items()}

    class TimeBudget(TrainerCallback):
        """Stop training once TIME_BUDGET_HOURS of wall-clock time are used."""

        def __init__(self):
            self.deadline = time.time() + TIME_BUDGET_HOURS * 3600

        def on_step_end(self, args, state, control, **k):
            """Request a stop after the current step once the deadline has passed."""
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control

    tok.padding_side = "right"
    # batch 4 x accumulation 4 = effective batch 16; a batch of 16 x 1024 tokens x 151k-token vocabulary
    # produced a 9 GB logits tensor and ran out of memory on a 24 GB L4.
    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collate, callbacks=[TimeBudget()])
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60

    tuned_res, tuned_samples = evaluate(model, tok, test, "LoRA fine-tuned")
    summary = {
        "model": MODEL,
        "train_examples": len(train_tok),
        "trainable_params": trainable,
        "total_params": total,
        "train_minutes": train_min,
        "final_train_loss": result.training_loss,
        "gpu": torch.cuda.get_device_name(0),
        "zero_shot": base_res,
        "fine_tuned": tuned_res,
    }
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump(
        {"zero_shot": base_samples, "fine_tuned": tuned_samples},
        open(os.path.join(OUT, "samples.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
