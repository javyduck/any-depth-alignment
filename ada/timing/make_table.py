"""Assemble the E6 inference-cost table from ``combined_timing_results.json``.

Reads the combined JSON produced by :mod:`ada.timing.benchmark` and renders the
paper's inference-cost table (§ E6): one row per guardrail (full forward pass)
and two rows per ADA-LP model (next-token and three-token KV-cached forwards).
Each cell is formatted as ``time_mean +/- time_std (memory_mean)`` with times in
milliseconds and memory in MB; out-of-memory cells are marked ``OOM``.

Run with::

    python -m ada.timing.make_table \
        --input timing_results/combined_timing_results.json \
        --output timing_results_table.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_timing_results(filepath: str) -> Dict[str, Any]:
    """Load the combined timing results JSON."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_time_memory(time_mean: float, time_std: float, memory_mean: float) -> str:
    """Format ``mean +/- std (memory)`` with times in ms and memory in MB."""
    time_mean_ms = time_mean * 1000
    time_std_ms = time_std * 1000
    return f"{time_mean_ms:.2f} ± {time_std_ms:.2f} ({memory_mean:.1f})"


def format_defense_time_memory(time_mean: float, time_std: float, memory_mean: float) -> str:
    """Like :func:`format_time_memory`, but renders OOM (negative time) as ``OOM``."""
    if time_mean < 0:
        return "OOM"
    time_mean_ms = time_mean * 1000
    time_std_ms = time_std * 1000
    return f"{time_mean_ms:.2f} ± {time_std_ms:.2f} ({memory_mean:.1f})"


def create_timing_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Build the inference-cost table DataFrame from the loaded results."""
    token_lengths = data["config"]["token_lengths"]
    results = data["results"]

    table_data = []

    for result in results:
        model_name = result["model_name"]
        token_data = result["token_lengths"]

        # Defense models expose forward_mean; ADA-LP models expose next_token_mean.
        sample_token_data = token_data[str(token_lengths[0])]
        is_defense_model = "forward_mean" in sample_token_data

        if is_defense_model:
            row_data = {"Model": model_name, "Type": "forward"}
            model_memory_cost = sample_token_data.get("model_memory_cost", 0)
            row_data["Model Memory Cost"] = f"{model_memory_cost:.1f}"

            for token_length in token_lengths:
                token_str = str(token_length)
                if token_str in token_data:
                    td = token_data[token_str]
                    row_data[f"{token_length}"] = format_defense_time_memory(
                        td.get("forward_mean", 0),
                        td.get("forward_std", 0),
                        td.get("forward_memory_mean", 0),
                    )
                else:
                    row_data[f"{token_length}"] = "N/A"

            table_data.append(row_data)

        else:
            row1_data = {"Model": model_name, "Type": "next_token", "Model Memory Cost": "0.0"}
            row2_data = {"Model": "", "Type": "forward_three_tokens", "Model Memory Cost": ""}

            for token_length in token_lengths:
                token_str = str(token_length)
                if token_str in token_data:
                    td = token_data[token_str]

                    next_token_mean = td.get("next_token_mean", 0)
                    next_token_std = td.get("next_token_std", 0)
                    next_token_memory_mean = td.get("next_token_memory_mean", 0)

                    forward_three_mean = td.get("forward_three_tokens_mean", 0)
                    forward_three_std = td.get("forward_three_tokens_std", 0)
                    forward_three_memory_mean = td.get("forward_three_tokens_memory_mean", 0)

                    # Approximate a missing next-token memory delta from the 3-token one.
                    if next_token_memory_mean < 0:
                        next_token_memory_mean = forward_three_memory_mean / 3

                    if next_token_mean < 0:
                        row1_data[f"{token_length}"] = "OOM"
                    else:
                        row1_data[f"{token_length}"] = format_time_memory(
                            next_token_mean, next_token_std, next_token_memory_mean
                        )

                    if forward_three_mean < 0:
                        row2_data[f"{token_length}"] = "OOM"
                    else:
                        row2_data[f"{token_length}"] = format_time_memory(
                            forward_three_mean, forward_three_std, forward_three_memory_mean
                        )
                else:
                    row1_data[f"{token_length}"] = "N/A"
                    row2_data[f"{token_length}"] = "N/A"

            table_data.append(row1_data)
            table_data.append(row2_data)

    columns = ["Model", "Type", "Model Memory Cost"] + [str(tl) for tl in token_lengths]
    return pd.DataFrame(table_data, columns=columns)


def print_table(df: pd.DataFrame) -> None:
    """Print the assembled table with explanatory units header."""
    print("Timing Results Table")
    print("=" * 120)
    print("Time units: milliseconds (ms)")
    print("Memory units: MB")
    print("Format: time_mean ± time_std (memory_mean) for all models")
    print("=" * 120)
    print()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)

    print(df.to_string(index=False))


def save_table_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write the assembled table to CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Table saved to: %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble the E6 inference-cost table.")
    parser.add_argument(
        "--input",
        type=str,
        default="timing_results/combined_timing_results.json",
        help="Path to combined_timing_results.json produced by ada.timing.benchmark.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="timing_results_table.csv",
        help="Output CSV path for the assembled table.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        logger.info("Loading timing results from %s", args.input)
        data = load_timing_results(args.input)

        logger.info("Creating timing table...")
        df = create_timing_table(data)

        print_table(df)
        save_table_to_csv(df, args.output)

    except FileNotFoundError:
        logger.error("Could not find file %s", args.input)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in file %s", args.input)


if __name__ == "__main__":
    main()
