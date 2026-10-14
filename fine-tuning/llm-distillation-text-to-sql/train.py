"""
Distilling a 7B LLM's text-to-SQL skill into a 0.5B model (sequence-level knowledge distillation)
==============================================================================================

QUESTION: Can a model 15x smaller learn a task from a big model's OUTPUTS alone -
no human labels - and how close does it get? This is how many production
teams cut LLM serving cost: use a large model as a teacher once, deploy a
small student forever.

SETUP (same data and execution-based metric as project 02):
    teacher : Qwen2.5-7B-Instruct, zero-shot, bf16 - writes SQL for 12k training questions
    student : Qwen2.5-0.5B-Instruct, full fine-tuning
    arms    : (a) student zero-shot
              (b) student trained on the TEACHER's SQL        <- distillation, no human labels
              (c) student trained on the teacher's SQL, keeping only queries that EXECUTE
                  without error (cheap automatic filtering - still no gold labels)
              (d) student trained on the human GOLD SQL        <- upper bound
METRICS: execution accuracy on 500 held-out questions (SQL is run in SQLite and
the result sets compared), plus generation throughput of teacher vs student.

Sequence-level KD (Kim & Rush, 2016): the student imitates the teacher's final
outputs rather than its per-token probabilities - which also works with a
teacher you can only call through an API.

PIPELINE (main):
    1. teacher answers the test set (its own score) and labels the training set
    2. student zero-shot score
    3. three fresh students, one per training-label source (arms b, c, d)
Every arm starts from the same pretrained student, sees the same questions
and is evaluated on the same test set - only the labels differ.

Pure helpers (prompt, SQL clean-up, execution matching) live in `common/sql.py`.
"""

import json
import os
import sys
import time

import torch

# Make the repo root importable so `common/` works from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.lm_data import masked_labels, pad_batch
from common.sql import clean_sql, execution_accuracy, run_sql
from common.sql import sql_messages as messages  # messages(question, context, answer=None)

QUICK = os.environ.get("QUICK") == "1"  # tiny smoke test
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.environ.get("OUT", "/content/sql_distill")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "6000"))  # teacher labelling is the expensive step
N_TEST = 40 if QUICK else 500


@torch.no_grad()  # inference only
def generate(model, tok, rows, bs=48, max_new=200):
    """Greedy-generate one SQL query per (question, context) pair in `rows`.

    Returns (cleaned_sql_list, {"seconds", "tokens_per_second"}) - the speed
    numbers are what makes the teacher-vs-student cost comparison concrete.
    """
    tok.padding_side = "left"  # batched generation with a decoder-only model needs left padding
    outs, t, n_tok = [], time.time(), 0
    for i in range(0, len(rows), bs):
        b = rows[i : i + bs]
        texts = [tok.apply_chat_template(messages(q, c), add_generation_prompt=True, tokenize=False) for q, c in b]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1] :]  # drop the prompt, keep generated tokens
        n_tok += int((new != tok.pad_token_id).sum())  # count real tokens (not padding after EOS)
        outs += [clean_sql(tok.decode(x, skip_special_tokens=True)) for x in new]
    secs = time.time() - t
    return outs, {"seconds": secs, "tokens_per_second": n_tok / secs}


def exec_accuracy(preds, rows):
    """Execution accuracy of `preds` against rows of (question, context, gold_sql)."""
    return execution_accuracy(preds, rows)


def finetune(name, pairs):
    """Full fine-tuning of the 0.5B student on (question, context, sql) triples; loss on the SQL only.

    Every call loads a *fresh* pretrained student, so the arms are independent.
    Returns (model, tokenizer, training_minutes).
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to("cuda")

    def enc(ex):
        """Tokenize prompt and prompt+answer; mask the prompt part of the labels with -100."""
        p = tok.apply_chat_template(messages(ex["q"], ex["c"]), add_generation_prompt=True, tokenize=False)
        f = tok.apply_chat_template(messages(ex["q"], ex["c"], ex["a"]), tokenize=False)
        pi, fi = tok(p, add_special_tokens=False)["input_ids"], tok(f, add_special_tokens=False)["input_ids"][:768]
        return {"input_ids": fi, "labels": masked_labels(pi, fi)}

    ds = Dataset.from_dict({"q": [x[0] for x in pairs], "c": [x[1] for x in pairs], "a": [x[2] for x in pairs]})
    ds = ds.map(enc, remove_columns=["q", "c", "a"])

    def collate(b):
        """Pad to the longest sequence in the batch and convert to tensors."""
        return {k: torch.tensor(v) for k, v in pad_batch(b, tok.pad_token_id).items()}

    tok.padding_side = "right"
    # Full fine-tuning (all 0.5B weights) -> a small LR (2e-5) to avoid destroying pretrained knowledge.
    args = TrainingArguments(
        output_dir=f"{OUT}/{name}",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    t = time.time()
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    return model, tok, (time.time() - t) / 60


def main():
    """Teacher labelling -> student baselines -> three distillation / gold arms -> metrics.json."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("gretelai/synthetic_text_to_sql")
    train = raw["train"].shuffle(seed=42).select(range(N_TRAIN))
    test = raw["test"].shuffle(seed=42).select(range(N_TEST * 3))
    # Keep only test questions whose gold SQL executes and returns rows (a meaningful comparison).
    test_rows = [(e["sql_prompt"], e["sql_context"], e["sql"]) for e in test if run_sql(e["sql_context"], e["sql"])][
        :N_TEST
    ]
    train_rows = [(e["sql_prompt"], e["sql_context"], e["sql"]) for e in train]
    res = {}

    # ---- teacher labels the training set (and is evaluated on the test set) ----
    ttok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to("cuda")
    t_test, t_speed = generate(teacher, ttok, [(q, c) for q, c, _ in test_rows])
    res["teacher_7b_zero_shot"] = {"execution_accuracy": exec_accuracy(t_test, test_rows), **t_speed}
    t = time.time()
    # The teacher writes SQL for every TRAINING question; the gold SQL is never shown to it.
    t_train, _ = generate(teacher, ttok, [(q, c) for q, c, _ in train_rows], bs=64)
    label_min = (time.time() - t) / 60
    del teacher  # free ~15 GB before loading students
    torch.cuda.empty_cache()
    # Automatic, label-free quality filter: does the teacher's SQL at least run?
    executes = [run_sql(c, a) is not None for (q, c, _), a in zip(train_rows, t_train)]
    print(f"teacher labelled {len(t_train)} questions in {label_min:.1f} min; {sum(executes)} of them execute")

    # ---- student arms ----
    stok = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to("cuda")
    s_test, s_speed = generate(student, stok, [(q, c) for q, c, _ in test_rows])
    res["student_0.5b_zero_shot"] = {"execution_accuracy": exec_accuracy(s_test, test_rows), **s_speed}
    del student

    # Same questions in every arm; only the source of the target SQL differs.
    arms = {
        "student_distilled_teacher_sql": [(q, c, a) for (q, c, _), a in zip(train_rows, t_train)],
        "student_distilled_executable_only": [
            (q, c, a) for (q, c, _), a, ok in zip(train_rows, t_train, executes) if ok
        ],
        "student_trained_on_gold_sql": train_rows,
    }
    for name, pairs in arms.items():
        model, tok, minutes = finetune(name, pairs)
        preds, speed = generate(model, tok, [(q, c) for q, c, _ in test_rows])
        res[name] = {
            "execution_accuracy": exec_accuracy(preds, test_rows),
            "train_examples": len(pairs),
            "train_minutes": minutes,
            **speed,
        }
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {
        "teacher": TEACHER,
        "student": STUDENT,
        "test_questions": len(test_rows),
        "teacher_labelling_minutes": label_min,
        "teacher_labels_executable": sum(executes) / len(executes),
        "gpu": torch.cuda.get_device_name(0),
        "results": res,
    }
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
