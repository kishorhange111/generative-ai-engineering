"""
Text-to-SQL helpers shared by `fine-tuning/qlora-text-to-sql` and
`fine-tuning/llm-distillation-text-to-sql`.

Why execution-based evaluation?
    Two SQL queries can look completely different and still return the same
    rows (aliases, column order, `IN (...)` vs. `OR`), and a query can look
    almost identical to the gold one and still be wrong. So instead of string
    matching we *run* both queries on the example's own tiny database and
    compare the result sets - the same idea as the Spider and BIRD benchmarks.

Every example in `gretelai/synthetic_text_to_sql` ships a `sql_context`
string with `CREATE TABLE ...; INSERT INTO ...;` statements, which is all we
need to build an in-memory SQLite database per example.
"""

import re
import sqlite3

# One system prompt for every model and every arm of every experiment, so the
# baseline and the fine-tuned model are compared on exactly the same input.
SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a question, reply with one SQLite query only."
)


def schema_only(context):
    """Keep only the CREATE TABLE statements of a `sql_context` string.

    The model sees the *schema* (tables and columns), not the data rows: in a
    real "chat with your database" product the model never gets to read the
    whole table either.

    >>> schema_only("CREATE TABLE t (a INT); INSERT INTO t VALUES (1);")
    'CREATE TABLE t (a INT);'
    """
    # Split on ";" into statements, keep those starting with CREATE and put the
    # terminating ";" back (split() removed it).
    return "\n".join(s.strip() + ";" for s in context.split(";") if s.strip().upper().startswith("CREATE"))


def sql_messages(question, context, answer=None):
    """Build the chat messages for one text-to-SQL example.

    Parameters
    ----------
    question : the natural-language question.
    context  : the example's `sql_context` (only its schema is shown).
    answer   : optional SQL string. When given it is appended as the assistant
               turn - used to build *training* sequences. When None the list
               ends with the user turn - used for *generation*.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_only(context)}\n\nQuestion: {question}"},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def run_sql(context, query):
    """Execute `query` against a fresh in-memory SQLite database built from `context`.

    Returns the result rows as a *sorted list of their repr() strings*, or
    None if anything fails (bad schema, syntax error, unknown column, ...).

    * sorted  -> the comparison ignores row order (no ORDER BY in most gold SQL).
    * repr()  -> rows of mixed types (int, float, str, None) become comparable
                 strings, so sorting never raises a TypeError.
    """
    con = sqlite3.connect(":memory:")  # a throw-away DB: examples never interfere
    try:
        con.executescript(context)  # CREATE TABLE + INSERT statements
        rows = con.execute(query).fetchall()
        return sorted(map(repr, rows))
    except Exception:
        # Any failure means "this SQL is not valid for this database".
        return None
    finally:
        con.close()


def clean_sql(text):
    """Turn raw model output into a single SQL statement ending in ';'.

    Models often wrap SQL in Markdown fences (```sql ... ```) or add a second
    statement / an explanation after the first ';'. We strip the fences and
    keep only the first statement.
    """
    # ^```(?:sql)?  -> an opening fence (optionally tagged "sql") at a line start
    # ```$          -> a closing fence at a line end
    text = re.sub(r"^```(?:sql)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    return text.split(";")[0].strip() + ";"


def normalize_sql(query):
    """Lower-case, collapse whitespace and drop a trailing ';' (for exact-match scoring)."""
    return " ".join(query.lower().split()).rstrip(";")


def is_exact_match(pred, gold):
    """Exact string match after `normalize_sql` - a strict, secondary metric."""
    return normalize_sql(pred) == normalize_sql(gold)


def execution_match(context, pred, gold):
    """True if `pred` runs without error AND returns the same rows as `gold`."""
    got = run_sql(context, pred)
    return got is not None and got == run_sql(context, gold)


def execution_accuracy(preds, rows):
    """Share of predictions whose results match the gold query.

    preds : list of predicted SQL strings.
    rows  : list of (question, context, gold_sql) tuples aligned with `preds`.
    """
    return sum(execution_match(c, p, gold) for p, (_q, c, gold) in zip(preds, rows)) / len(rows)
