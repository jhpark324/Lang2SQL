"""Pair structured SQL review with the published unstructured #290 run."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from bench.bird.dataset import verify_databases
from bench.bird.protocol import BASELINE as REVIEW_PRODUCT
from bench.common.io import digest, json_digest, read_json, write_json

REFERENCE_PR_HEAD = "151939f9adcee2c073832eaf5df0e161cf73aa52"
REFERENCE_PRODUCT = "0a7d85d8155b7202d45999631f526e028ddc1e1e"
REFERENCE_CALL_LIMIT, REVIEW_CALL_LIMIT = 8, 12
MATCHED_SETTINGS = (
    "mode",
    "condition",
    "profile",
    "manifest_digest",
    "runtime",
    "provider_options",
    "scoring",
    "question_order",
    "seed",
    "instruction",
    "evidence_label",
    "term_custom",
    "lock_sha256",
    "evaluator_source",
    "tools",
)


def candidate_sql(arguments):
    sql = arguments.get("sql") if isinstance(arguments, dict) else None
    return sql.strip() if isinstance(sql, str) else None


def review_metrics(record):
    """Match candidates and separate assistant responses using recorded tool IDs."""
    calls = record["calls"]
    auxiliary = [call for call in calls if not call["harness"]]
    requests = {
        request["id"]: (turn, request["name"])
        for turn, call in enumerate(calls)
        if call["harness"]
        for request in call.get("tool_calls", [])
    }
    executed = {event["call_id"] for event in record["recovery"]["db_executions"]}
    submitted = record.get("sql")
    verdicts = Counter()
    errors = Counter()
    latest = {}
    final_reviewed = final_passed = final_ran = final_pass_then_ran = False
    for tool in record["tools"]:
        call_id = tool["call_id"]
        turn, name = requests[call_id]
        sql = candidate_sql(tool["arguments"])
        if name == "review_sql":
            verdict = (
                "error" if tool["is_error"] else json.loads(tool["content"])["verdict"]
            )
            verdicts[verdict] += 1
            if tool["is_error"]:
                errors["invalid_verdict"] += tool["content"].startswith(
                    "review_sql returned no valid verdict."
                )
                errors["budget_denied"] += tool["content"].startswith(
                    "CallBudgetExhausted:"
                )
            if sql:
                latest[sql] = (turn, verdict)
                if sql == submitted:
                    final_reviewed |= verdict in {"pass", "revise"}
                    final_passed |= verdict == "pass"
        elif name == "run_sql" and submitted and sql == submitted:
            if not tool["is_error"] and call_id in executed:
                final_ran = True
                prior_turn, verdict = latest.get(sql, (turn, None))
                final_pass_then_ran |= prior_turn < turn and verdict == "pass"

    requested = sum(name == "review_sql" for _, name in requests.values())
    return {
        "requests": requested,
        "dispatches": sum(verdicts.values()),
        "pass": verdicts["pass"],
        "revise": verdicts["revise"],
        "error": verdicts["error"],
        "invalid_verdict": errors["invalid_verdict"],
        "budget_denied": errors["budget_denied"],
        "api_calls": len(auxiliary),
        "structured_api_calls": sum("response_format" in call for call in auxiliary),
        "api_errors": sum(call["status"] == "error" for call in auxiliary),
        "api_sql_fences": sum(
            bool(re.search(r"```sql\b", call.get("content", ""), re.IGNORECASE))
            for call in auxiliary
        ),
        "api_seconds": sum(call["seconds"] for call in auxiliary),
        "no_review_request": int(requested == 0),
        "submitted_without_review_request": int(bool(submitted) and requested == 0),
        "submitted_with_matching_verdict": int(final_reviewed),
        "submitted_with_matching_pass": int(final_passed),
        "submitted_with_matching_execution": int(final_ran),
        "submitted_with_pass_then_execution": int(final_pass_then_ran),
    }


def archive_records(destination, records):
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as archive:
            for record in records:
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                archive.write((line + "\n").encode("utf-8"))


def main():
    # Import the aggregation used by #280 from its pinned worktree on PYTHONPATH.
    from bench.report_schema_recovery import (
        case_metrics,
        index,
        read_lines,
        require,
        summarize,
        write_lines,
    )
    from lang2sql.adapters.llm.openai_ import _encode_response_format
    from lang2sql.tools.review_sql import _VERDICT_SCHEMA

    response_format = _encode_response_format("sql_review_verdict", _VERDICT_SCHEMA)

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "run", "db-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.reference / "manifest.json"
    manifest = read_json(manifest_path)
    databases = verify_databases(args.db_dir, manifest)
    cases = index(manifest["cases"], "question_id")
    reference = read_json(args.reference / "comparison.json")
    require(reference["treatment_product"] == REFERENCE_PRODUCT, "Wrong #290 arm")
    require(reference["manifest_sha256"] == digest(manifest_path), "Wrong manifest")
    require(
        len(cases) == 365 and len(reference["groups"]) == 9, "Wrong evaluation matrix"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, args.output / "manifest.json")

    groups = {}
    for name, previous in reference["groups"].items():
        baseline = args.reference / name
        run = args.run / name
        before_settings = read_json(baseline / "settings.json")
        settings = read_json(run / "settings.json")
        status = read_json(run / "status.json")
        summary = read_json(run / "summary.json")
        require(
            before_settings["product_commit"] == REFERENCE_PRODUCT,
            f"Wrong base: {name}",
        )
        require(settings["product_commit"] == REVIEW_PRODUCT, f"Wrong product: {name}")
        require(
            settings["manifest_digest"] == json_digest(manifest), f"Wrong input: {name}"
        )
        require(
            (run / "manifest.json").read_bytes() == manifest_path.read_bytes(),
            f"Wrong manifest: {name}",
        )
        for key in MATCHED_SETTINGS:
            require(before_settings[key] == settings[key], f"Changed {key}: {name}")
        require(
            before_settings["budgets"]["max_calls"] == REFERENCE_CALL_LIMIT,
            f"Unexpected #290 call limit: {name}",
        )
        require(
            settings["budgets"]
            == {**before_settings["budgets"], "max_calls": REVIEW_CALL_LIMIT},
            f"Unexpected budget changes: {name}",
        )
        old_code = before_settings["code_sha256"]
        new_code = settings["code_sha256"]
        require(old_code.keys() == new_code.keys(), f"Evaluator files changed: {name}")
        changed_code = {key for key in old_code if old_code[key] != new_code[key]}
        require(
            changed_code == {"bench/bird/protocol.py", "bench/common/providers.py"},
            f"Unexpected evaluator diff: {name}",
        )
        require(
            status["status"] == summary["status"] == "completed",
            f"Incomplete run: {name}",
        )
        require(
            status["completed"] == status["expected"] == summary["total"] == len(cases),
            f"Wrong count: {name}",
        )
        require(
            digest(run / "scores.json") == summary["scores_sha256"],
            f"Score digest mismatch: {name}",
        )
        provenance = read_json(baseline / "provenance.json")
        for filename in ("settings.json", "scores.json"):
            require(
                digest(baseline / filename)
                == provenance[f"{Path(filename).stem}_sha256"],
                f"Wrong #290 {filename}: {name}",
            )

        base_rows = index(read_lines(baseline / "paired-scores.jsonl"), "question_id")
        base_scores = index(read_json(baseline / "scores.json"), "question_id")
        scores = index(read_json(run / "scores.json"), "question_id")
        require(
            cases.keys() == base_rows.keys() == base_scores.keys() == scores.keys(),
            f"Pairing mismatch: {name}",
        )
        require(
            {path.stem for path in (run / "records").glob("*.json")}
            == {str(ident) for ident in cases},
            f"Record set mismatch: {name}",
        )
        settings_digest = json_digest(settings)
        records = []
        rows = []
        for ident, case in cases.items():
            result = read_json(run / "records" / f"{ident}.json")
            score = scores[ident]
            before = base_rows[ident]["treatment"]
            require(
                ident == result["question_id"]
                and case["db_id"] == score["db_id"] == base_scores[ident]["db_id"],
                f"Case identity mismatch: {name}/{ident}",
            )
            require(
                base_scores[ident]["ex"] == before["correct"],
                f"Prior score mismatch: {name}/{ident}",
            )
            require(
                json_digest(result) == score["record_digest"],
                f"Record digest mismatch: {name}/{ident}",
            )
            require(
                result["profile"] == score["profile"] == settings["profile"]["name"],
                f"Wrong profile: {name}/{ident}",
            )
            require(
                result["condition"] == score["condition"] == settings["condition"],
                f"Wrong evidence: {name}/{ident}",
            )
            require(
                result["settings_digest"]
                == summary["settings_digest"]
                == settings_digest,
                f"Wrong run identity: {name}/{ident}",
            )
            require(
                len(result["calls"]) <= settings["budgets"]["max_calls"],
                f"Over budget: {name}/{ident}",
            )
            require(
                all(
                    call.get("response_format")
                    == (None if call["harness"] else response_format)
                    for call in result["calls"]
                ),
                f"Unexpected response format: {name}/{ident}",
            )
            after = case_metrics(
                "bird",
                result,
                score,
                result.get("sql"),
                databases[case["db_id"]],
                baseline=False,
            )
            after["review"] = review_metrics(result)
            after["exceeded_previous_limit"] = (
                len(result["calls"]) > REFERENCE_CALL_LIMIT
            )
            after["reached_call_limit"] = len(result["calls"]) == REVIEW_CALL_LIMIT
            rows.append(
                {
                    "question_id": ident,
                    "db_id": case["db_id"],
                    "baseline": before,
                    "treatment": after,
                }
            )
            records.append(result)

        interrupted = [
            read_json(path) for path in sorted(run.glob("interruptions/*/*.json"))
        ]
        before = previous["treatment"]
        after = summarize([row["treatment"] for row in rows], records, interrupted)
        require(
            before["correct"] == sum(row["baseline"]["correct"] for row in rows),
            f"Wrong prior total: {name}",
        )
        require(after["correct"] == summary["correct"], f"Wrong total: {name}")
        model = settings["profile"]["model"].split("/", 1)[1]
        require(
            set(after["response_models"]) == set(before["response_models"]) == {model},
            f"Response model mismatch: {name}",
        )
        review = Counter()
        transitions = {}
        paired = Counter()
        call_limit = Counter()
        for row in rows:
            left, right = row["baseline"], row["treatment"]
            review.update(right["review"])
            transitions.setdefault(left["state"], Counter())[right["state"]] += 1
            paired[left["correct"], right["correct"]] += 1
            past_limit = right["exceeded_previous_limit"]
            call_limit.update(
                cases_above_previous_limit=int(past_limit),
                correct_above_previous_limit=int(past_limit and right["correct"]),
                reached_new_limit=int(right["reached_call_limit"]),
                previously_exhausted_now_correct=int(
                    left["budget_without_sql"] and right["correct"]
                ),
            )
        after.update(
            accuracy_pct=100 * after["correct"] / len(rows), review=dict(review)
        )
        groups[name] = {
            "profile": settings["profile"]["name"],
            "condition": settings["condition"],
            "requested_model": settings["profile"]["model"],
            "pairs": len(rows),
            "baseline": before,
            "treatment": after,
            "baseline_only": paired[1, 0],
            "treatment_only": paired[0, 1],
            "both_correct": paired[1, 1],
            "both_wrong": paired[0, 0],
            "delta_percentage_points": (after["correct"] - before["correct"])
            * 100
            / len(rows),
            "state_transitions": transitions,
            "call_limit": dict(call_limit),
        }
        export = args.output / name
        write_lines(export / "paired-scores.jsonl", rows)
        archive_records(export / "records.jsonl.gz", records)
        write_lines(export / "interruptions.jsonl", interrupted)
        for filename in ("settings.json", "scores.json", "status.json", "summary.json"):
            shutil.copyfile(run / filename, export / filename)
        write_json(
            export / "provenance.json",
            {
                "baseline_settings_sha256": provenance["settings_sha256"],
                "baseline_scores_sha256": provenance["scores_sha256"],
                "settings_sha256": digest(export / "settings.json"),
                "scores_sha256": digest(export / "scores.json"),
                "records_gzip_sha256": digest(export / "records.jsonl.gz"),
            },
        )

    write_json(
        args.output / "comparison.json",
        {
            "benchmark": "BIRD Mini-Dev SQLite",
            "baseline_pr_head": REFERENCE_PR_HEAD,
            "baseline_product": REFERENCE_PRODUCT,
            "treatment_product": REVIEW_PRODUCT,
            "baseline_max_calls": REFERENCE_CALL_LIMIT,
            "treatment_max_calls": REVIEW_CALL_LIMIT,
            "manifest_sha256": digest(manifest_path),
            "paired_cases": sum(group["pairs"] for group in groups.values()),
            "groups": groups,
        },
    )


if __name__ == "__main__":
    main()
