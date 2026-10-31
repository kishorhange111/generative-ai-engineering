"""Tests for the text-to-SQL prompt, SQL clean-up and execution-based matching."""

from genai_common.sql import (
    SYSTEM_PROMPT,
    clean_sql,
    execution_accuracy,
    execution_match,
    is_exact_match,
    normalize_sql,
    run_sql,
    schema_only,
    sql_messages,
)

CTX = (
    "CREATE TABLE sales (id INT, country TEXT, qty INT);"
    "INSERT INTO sales VALUES (1, 'UK', 3), (2, 'DE', 5), (3, 'UK', 4);"
)


def test_schema_only_keeps_create_statements_only():
    assert schema_only(CTX) == "CREATE TABLE sales (id INT, country TEXT, qty INT);"


def test_schema_only_handles_multiple_tables_and_case():
    ctx = "create table a (x INT); INSERT INTO a VALUES (1); CREATE TABLE b (y INT);"
    assert schema_only(ctx) == "create table a (x INT);\nCREATE TABLE b (y INT);"


def test_sql_messages_without_answer_ends_with_user_turn():
    msgs = sql_messages("How many?", CTX)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "INSERT" not in msgs[1]["content"]  # data rows are never shown to the model
    assert msgs[1]["content"].endswith("Question: How many?")


def test_sql_messages_with_answer_appends_assistant_turn():
    msgs = sql_messages("How many?", CTX, "SELECT 1;")
    assert msgs[-1] == {"role": "assistant", "content": "SELECT 1;"}


def test_run_sql_returns_sorted_rows():
    assert run_sql(CTX, "SELECT qty FROM sales WHERE country = 'UK' ORDER BY qty DESC") == ["(3,)", "(4,)"]


def test_run_sql_returns_none_on_error():
    assert run_sql(CTX, "SELECT nope FROM sales") is None
    assert run_sql(CTX, "SELEC qty FROM sales") is None


def test_run_sql_isolated_between_calls():
    run_sql(CTX, "DELETE FROM sales")
    assert len(run_sql(CTX, "SELECT * FROM sales")) == 3


def test_clean_sql_strips_fences_and_extra_statements():
    assert clean_sql("```sql\nSELECT * FROM t;\n```") == "SELECT * FROM t;"
    assert clean_sql("SELECT 1; SELECT 2;") == "SELECT 1;"
    assert clean_sql("  SELECT 1  ") == "SELECT 1;"


def test_normalize_and_exact_match():
    assert normalize_sql("SELECT  *\nFROM t;") == "select * from t"
    assert is_exact_match("select * from T", "SELECT *   FROM t;")
    assert not is_exact_match("SELECT a FROM t", "SELECT b FROM t")


def test_execution_match_accepts_equivalent_queries():
    gold = "SELECT SUM(qty) FROM sales WHERE country = 'UK'"
    pred = "SELECT SUM(s.qty) AS total FROM sales s WHERE s.country IN ('UK')"
    assert execution_match(CTX, pred, gold)


def test_execution_match_rejects_wrong_rows_and_errors():
    gold = "SELECT SUM(qty) FROM sales WHERE country = 'UK'"
    assert not execution_match(CTX, "SELECT SUM(qty) FROM sales", gold)
    assert not execution_match(CTX, "SELECT broken FROM", gold)


def test_execution_match_is_order_insensitive():
    assert execution_match(CTX, "SELECT id FROM sales ORDER BY id DESC", "SELECT id FROM sales")


def test_execution_accuracy():
    rows = [("q1", CTX, "SELECT COUNT(*) FROM sales"), ("q2", CTX, "SELECT MAX(qty) FROM sales")]
    assert execution_accuracy(["SELECT COUNT(id) FROM sales", "SELECT MIN(qty) FROM sales"], rows) == 0.5
