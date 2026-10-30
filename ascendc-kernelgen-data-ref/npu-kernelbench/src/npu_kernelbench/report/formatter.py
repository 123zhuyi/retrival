"""Result formatting — Rich table output for evaluation traces."""

from rich.console import Console
from rich.table import Table

from npu_kernelbench.data.trace import Trace, EvaluationStatus

console = Console(stderr=True)


def format_traces_table(traces: list[Trace], title: str = "Evaluation Results") -> Table:
    """Build a Rich table from a list of evaluation traces."""
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=22)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Ref (ms)", justify="right", width=14)
    table.add_column("Speedup", justify="right", width=10)
    table.add_column("Max Abs Err", justify="right", width=14)
    table.add_column("Max Rel Err", justify="right", width=14)

    passed = 0
    total = len(traces)
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            table.add_row(str(i), "[dim]no evaluation[/dim]", "", "", "", "", "")
            continue

        status = ev.status.value
        if ev.status == EvaluationStatus.PASSED:
            status_str = f"[green]{status}[/green]"
            passed += 1
        elif ev.status == EvaluationStatus.INCORRECT_NUMERICAL:
            status_str = f"[yellow]{status}[/yellow]"
        elif ev.status == EvaluationStatus.REWARD_HACK:
            status_str = f"[magenta]{status}[/magenta]"
        else:
            status_str = f"[red]{status}[/red]"

        latency = ""
        ref_latency = ""
        speedup = ""
        if ev.performance:
            latency = f"{ev.performance.latency_ms:.3f}"
            ref_latency = f"{ev.performance.reference_latency_ms:.3f}"
            speedup = f"{ev.performance.speedup_factor:.2f}x"

        abs_err = ""
        rel_err = ""
        if ev.correctness:
            if ev.correctness.has_nan:
                abs_err = "NaN"
                rel_err = "NaN"
            elif ev.correctness.has_inf:
                abs_err = "Inf"
                rel_err = "Inf"
            else:
                abs_err = f"{ev.correctness.max_absolute_error:.2e}"
                rel_err = f"{ev.correctness.max_relative_error:.2e}"

        table.add_row(
            str(i), status_str, latency, ref_latency, speedup, abs_err, rel_err
        )

    console.print(table)
    console.print(f"\n[bold]{passed}/{total}[/bold] workloads passed")

    _print_error_logs(traces)

    return table


def _print_error_logs(traces: list[Trace]) -> None:
    """Print runtime logs for traces with errors."""
    error_logs = [
        (i, ev.status.value, ev.log)
        for i, trace in enumerate(traces)
        if (ev := trace.evaluation)
        and ev.status
        not in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL)
        and ev.log
    ]
    if error_logs:
        console.print(f"\n[bold red]Runtime logs ({len(error_logs)}):[/bold red]")
        for idx, status, log in error_logs:
            console.print(f"\n[bold]Workload {idx}[/bold] ([red]{status}[/red]):")
            console.print(log.rstrip())
