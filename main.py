import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from negmas import make_issue
from negmas.inout import Scenario
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
from negmas.preferences.value_fun import AffineFun, IdentityFun
from negmas.sao.negotiators import BoulwareTBNegotiator as Boulware
from negmas.sao.negotiators import LinearTBNegotiator as Linear
from negmas.sao.negotiators import NaiveTitForTatNegotiator
from negmas.tournaments.neg import cartesian_tournament

from Group34_Negotiator import Group34_Negotiator


SCENARIO_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path.home() / "negmas_results_parallel"
COMPETITORS = [Group34_Negotiator, Boulware, Linear, NaiveTitForTatNegotiator]
TOURNAMENT_SETTINGS = {
    "n_repetitions": 2,
    "mechanism_params": {"n_steps": 50, "time_limit": 2},
}
MAX_WORKERS = 4


def build_value_function(spec):
    kind = spec["type"]
    if kind == "identity":
        return IdentityFun()
    if kind == "affine":
        return AffineFun(
            slope=spec.get("slope", 1.0),
            bias=spec.get("bias", 0.0),
        )
    raise ValueError(f"Unsupported value function type: {kind}")


<<<<<<< HEAD
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=[scenario],
        n_repetitions=5,
        path=path,
    )
=======
def build_issue(issue_spec):
    values = issue_spec["values"]
    if isinstance(values, list):
        values = tuple(values)
    return make_issue(name=issue_spec["name"], values=values)
>>>>>>> d4d12d2e3dd1d49dd04c6392a10cc249a6e4c3a3


def load_scenario(scenario_file: Path):
    with scenario_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    issues = [build_issue(issue_spec) for issue_spec in data["issues"]]
    outcome_space = make_os(issues=issues)

    ufuns = []
    for agent_spec in data["agents"]:
        ufun = UFun(
            values=[build_value_function(value_spec) for value_spec in agent_spec["values"]],
            outcome_space=outcome_space,
            reserved_value=agent_spec.get("reserved_value", 0.0),
            weights=agent_spec.get("weights"),
        )
        ufuns.append(ufun.normalize())

    return data["name"], Scenario(outcome_space=outcome_space, ufuns=ufuns)


def run_scenario_tournament(scenario_file: str):
    scenario_path = Path(scenario_file)
    scenario_name, scenario = load_scenario(scenario_path)
    results_path = RESULTS_DIR / scenario_path.stem
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            results = cartesian_tournament(
                competitors=COMPETITORS,
                scenarios=[scenario],
                path=results_path,
                **TOURNAMENT_SETTINGS,
            )
    return {
        "scenario": scenario_name,
        "file": scenario_path.name,
        "scores": results.final_scores.sort_values("score", ascending=False).to_dict("records"),
    }


def discover_scenario_files():
    return sorted(SCENARIO_DIR.glob("*.json"))


def format_scores(rows):
    if not rows:
        return ["  No scores available"]

    best_score = rows[0]["score"]
    lines = []
    for rank, row in enumerate(rows, start=1):
        gap = best_score - row["score"]
        lines.append(
            f"  {rank}. {row['strategy']:<24} score={row['score']:.4f}  gap={gap:.4f}"
        )
    return lines


def build_overall_summary(results):
    summary = {}
    for result in results:
        for rank, row in enumerate(result["scores"], start=1):
            strategy = row["strategy"]
            stats = summary.setdefault(
                strategy,
                {"total_score": 0.0, "total_rank": 0, "wins": 0, "scenarios": 0},
            )
            stats["total_score"] += row["score"]
            stats["total_rank"] += rank
            stats["wins"] += int(rank == 1)
            stats["scenarios"] += 1

    overall_rows = []
    for strategy, stats in summary.items():
        scenarios = max(1, stats["scenarios"])
        overall_rows.append(
            {
                "strategy": strategy,
                "avg_score": stats["total_score"] / scenarios,
                "avg_rank": stats["total_rank"] / scenarios,
                "wins": stats["wins"],
                "scenarios": scenarios,
            }
        )

    overall_rows.sort(key=lambda row: (-row["avg_score"], row["avg_rank"], -row["wins"]))
    return overall_rows


def format_overall_summary(rows):
    lines = ["\nOverall Summary:"]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "  "
            f"{rank}. {row['strategy']:<24} avg_score={row['avg_score']:.4f}  "
            f"avg_rank={row['avg_rank']:.2f}  wins={row['wins']}/{row['scenarios']}"
        )
    return lines


def write_summary(results, overall_rows):
    summary_path = RESULTS_DIR / "summary.json"
    payload = {
        "per_scenario": results,
        "overall": overall_rows,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return summary_path


if __name__ == "__main__":
    scenario_files = discover_scenario_files()
    if not scenario_files:
        raise SystemExit(f"No scenario files found in {SCENARIO_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    collected_results = []

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(scenario_files))) as executor:
        futures = {
            executor.submit(run_scenario_tournament, str(scenario_file)): scenario_file
            for scenario_file in scenario_files
        }

        for future in as_completed(futures):
            result = future.result()
            collected_results.append(result)
            print(f"\nScenario: {result['scenario']} ({result['file']})")
            for line in format_scores(result["scores"]):
                print(line)

    overall_rows = build_overall_summary(collected_results)
    for line in format_overall_summary(overall_rows):
        print(line)
    summary_path = write_summary(collected_results, overall_rows)
    print(f"\nSaved summary to {summary_path}")
