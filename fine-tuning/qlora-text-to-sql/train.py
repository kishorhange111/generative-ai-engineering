"""
QLoRA fine-tuning of a 7B LLM for text-to-SQL, evaluated by *executing* the SQL
===============================================================================

TASK: "Which customers from Germany spent more than 500 last year?"  +  table schema
      ->  SELECT ... FROM ... WHERE ...
Natural-language access to databases is one of the most requested enterprise
LLM features (analytics copilots, "chat with your data", MCP database tools).

QLoRA (Dettmers et al., 2023) - fine-tune a 7-billion-parameter model on one GPU:
    * The base model is loaded in 4-bit NF4 ("NormalFloat4", an information-
      theoretically optimal 4-bit type for normally distributed weights) with
      double quantization -> 7B weights take ~5 GB instead of ~15 GB (bf16).
    * The 4-bit weights are FROZEN. Small LoRA adapters (rank 16) are added to
      every linear layer of every Transformer block and trained in bf16.
    * Only ~0.5% of parameters receive gradients/optimizer state.
    * Paged optimizers + gradient checkpointing keep memory spikes in check.

EVALUATION - execution accuracy, not string matching:
    Every test example ships with CREATE TABLE + INSERT statements. We build the
    database in SQLite, run the gold query and the model's query, and compare
    result sets. Two different-looking queries that return the same rows are
    both correct; a query that looks right but returns the wrong rows is wrong.
    The same metric is used by the Spider and BIRD text-to-SQL benchmarks.
    Compared: the base instruct model (zero-shot, same prompt) vs. QLoRA-tuned.

Data: gretelai/synthetic_text_to_sql - 105k examples across 100 domains.

PIPELINE (main):
    1. load data, keep only test examples whose gold SQL actually runs
    2. load the base model in 4-bit and evaluate it zero-shot   (baseline)
    3. attach LoRA adapters and train for one epoch             (QLoRA)
    4. evaluate again with the same prompt, save metrics + adapter

The pure helpers (prompt, SQL clean-up, execution matching) live in
`common/sql.py` and are unit-tested in `tests/test_sql.py`.
"""
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Make the repo root importable so `common/` works when the script is launched
# as `python fine-tuning/qlora-text-to-sql/train.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.lm_data import masked_labels, pad_batch
from common.sql import clean_sql, is_exact_match, run_sql, sql_messages

# ---------------------------------------------------------------------------
# Configuration (environment variables let one script serve smoke tests and full runs)
# ---------------------------------------------------------------------------
QUICK = os.environ.get("QUICK") == "1"                         # tiny smoke test before spending GPU hours
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = os.environ.get("OUT", "/content/qlora_sql")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "12000"))
N_TEST = 40 if QUICK else int(os.environ.get("N_TEST", "500"))
MAX_LEN = 768                                                    # tokens; schemas + SQL fit comfortably
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.2"))   # hard cap on paid GPU time
# bf16 has the same exponent range as fp32 (no overflow / loss scaling needed),
# but only Ampere+ GPUs (A100, L4) support it; fall back to fp16 on older cards (T4).
BF16 = torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16 else torch.float16


# ---------------------------------------------------------------------------
# Data + prompt
# ---------------------------------------------------------------------------
def messages(ex, with_answer):
    """Chat messages for a dataset row; the gold SQL is appended only when `with_answer`."""
    return sql_messages(ex["sql_prompt"], ex["sql_context"], ex["sql"] if with_answer else None)


def executable(ex):
    """Keep test examples whose gold SQL runs in SQLite and returns something (so the test is meaningful).

    A gold query that errors or returns no rows would make "correct" trivially
    easy (any query returning nothing would match), so such rows are dropped.
    """
    rows = run_sql(ex["sql_context"], ex["sql"])
    return bool(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()   # no gradients needed for inference -> less memory, faster
def evaluate(model, tok, test, label):
    """Generate SQL for every test question (greedy) and score it.

    Returns (metrics, samples):
        execution_accuracy - result rows equal the gold query's rows  (main metric)
        valid_sql_rate     - the query runs without error
        exact_match        - normalised SQL string equals the gold string (strict)
    and up to 12 example predictions for the README / error analysis.
    """
    model.eval()                          # disables dropout
    tok.padding_side = "left"             # decoder-only models must be left-padded for batched generation
    correct = valid = exact = 0
    samples = []
    start = time.time()
    for i in range(0, len(test), 16):     # batches of 16 questions
        batch = test.select(range(i, min(i + 16, len(test))))
        # add_generation_prompt=True appends "<|im_start|>assistant\n" so the model starts its answer.
        prompts = [tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False) for ex in batch]
        # add_special_tokens=False: the chat template already contains every special token.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # Greedy decoding (do_sample=False) makes the evaluation deterministic.
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        # generate() returns prompt + continuation; slice off the prompt to keep only the new tokens.
        for ex, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            pred = clean_sql(tok.decode(g, skip_special_tokens=True))
            got, gold = run_sql(ex["sql_context"], pred), run_sql(ex["sql_context"], ex["sql"])
            valid += got is not None
            correct += got is not None and got == gold
            exact += is_exact_match(pred, ex["sql"])
            if len(samples) < 12:
                samples.append({"question": ex["sql_prompt"], "gold": ex["sql"], "pred": pred, "correct": got == gold})
    n = len(test)
    res = {"execution_accuracy": correct / n, "valid_sql_rate": valid / n, "exact_match": exact / n,
           "eval_minutes": (time.time() - start) / 60}
    print(label, res, flush=True)
    return res, samples


class TimeBudget(TrainerCallback):
    """Stop training cleanly once `hours` of wall-clock time have passed (caps the GPU bill)."""

    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kw):
        """Called by the Trainer after every optimizer step; request a stop past the deadline."""
        if time.time() > self.deadline:
            print(f"[time budget] stopping at step {state.global_step}")
            control.should_training_stop = True
        return control


def main():
    """Baseline evaluation -> QLoRA training -> evaluation -> save metrics, samples and adapter."""
    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("gretelai/synthetic_text_to_sql")
    # Fixed seeds -> the same train/test examples on every run.
    train = raw["train"].shuffle(seed=42).select(range(N_TRAIN))
    # Over-sample 3x, then keep only executable examples, then cut to N_TEST.
    test = raw["test"].shuffle(seed=42).select(range(N_TEST * 3)).filter(executable)
    test = test.select(range(min(N_TEST, len(test))))
    print(f"train {len(train)}, executable test {len(test)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:             # some tokenizers ship without a pad token
        tok.pad_token = tok.eos_token

    # 4-bit NF4 quantization with double quantization; matmuls computed in bf16 (fp16 on older GPUs).
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=DTYPE)
    # device_map={"": 0}: put the whole model on GPU 0 (no CPU offload).
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, dtype=DTYPE, device_map={"": 0})
    print(f"GPU memory after loading 4-bit model: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    # ---- Baseline: zero-shot base model ----
    base_res, base_samples = evaluate(model, tok, test, "base (zero-shot)")

    # ---- QLoRA ----
    # Casts norms to fp32 for stability and enables gradient checkpointing
    # (recompute activations in the backward pass instead of storing them).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # r=16: rank of the low-rank update B @ A; lora_alpha/r = 2 scales it.
    # Adapters on every attention (q/k/v/o) and MLP (gate/up/down) projection.
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    tok.padding_side = "right"            # training: pad on the right (labels are padded with -100 anyway)

    def encode(ex):
        """Loss only on the SQL answer: prompt tokens get label -100."""
        prompt = tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(messages(ex, True), tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        return {"input_ids": f_ids, "labels": masked_labels(p_ids, f_ids)}

    train_tok = train.map(encode, remove_columns=train.column_names)

    def collate(batch):
        """Pad a batch to its longest sequence and convert to tensors."""
        return {k: torch.tensor(v) for k, v in pad_batch(batch, tok.pad_token_id).items()}

    args = TrainingArguments(
        # Effective batch = 8 x 2 = 16 sequences per optimizer step.
        output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        # 2e-4 is the usual LoRA learning rate (adapters start at zero, so they need a larger LR
        # than full fine-tuning); max_grad_norm 0.3 follows the QLoRA paper.
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05, max_grad_norm=0.3,
        # Paged 8-bit AdamW: optimizer states in 8 bit, paged to CPU memory on spikes.
        optim="paged_adamw_8bit", bf16=BF16, fp16=not BF16, logging_steps=10, save_strategy="no",
        # remove_unused_columns=False: our collate builds the model inputs itself.
        remove_unused_columns=False, report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collate,
                      callbacks=[TimeBudget(TIME_BUDGET_HOURS)])
    torch.cuda.reset_peak_memory_stats()  # measure peak memory of training alone
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    # Gradient checkpointing disables the KV cache; turn it back on for fast generation.
    model.config.use_cache = True
    tuned_res, tuned_samples = evaluate(model, tok, test, "QLoRA fine-tuned")
    model.save_pretrained(os.path.join(OUT, "adapter"))   # only the LoRA weights (~150 MB), not the 7B base

    summary = {"model": MODEL, "train_examples": len(train), "test_examples": len(test),
               "trainable_params": trainable, "total_params": total, "trainable_percent": 100 * trainable / total,
               "train_minutes": train_min, "peak_train_gpu_memory_gb": peak_gb,
               "final_train_loss": result.training_loss, "gpu": torch.cuda.get_device_name(0),
               "base": base_res, "qlora": tuned_res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"base": base_samples, "qlora": tuned_samples}, open(os.path.join(OUT, "samples.json"), "w"), indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
