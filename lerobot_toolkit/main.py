"""LeRobot Dataset Toolkit — CLI entry point."""

import typer
import os
import json
from rich.console import Console
from rich.table import Table
from src.metadata import load_info, load_tasks, load_episodes
from src.episodes import calculate_avg_stats
from src.episodes import list_episodes_summary
from src.checker import run_checks, write_check_report, print_rich_summary
from src.report import generate_report
from src.replay import replay_episode

app = typer.Typer(
    name="lerobot_toolkit",
    help="A toolkit for inspecting, analyzing, and vali`dating LeRobot datasets.",
)

@app.command()
def info(dataset: str=typer.Option(..., "--dataset", help="Path to the dataset directory."), json_output: bool = typer.Option(False, "--json", help="Output metadata as JSON.")):
    """Display dataset metadata and structure."""
    
    meta = load_info(dataset)
    episodes = load_episodes(dataset)
    tasks = load_tasks(dataset)

    summary = calculate_avg_stats(episodes, meta["fps"])

    cameras = [
        k for k in meta["features"].keys()
        if k.startswith("observation.images.")
    ]
    
    table = Table(title="[bold red]LeRobot Dataset Info[/bold red]")

    table.add_column("Property", header_style="yellow bold", justify = "center", style="cyan", no_wrap=True)
    table.add_column("Value", header_style="yellow bold", justify = "left")

    table.add_row("Dataset Path", os.path.abspath(dataset))
    table.add_row("Codebase Version", meta["codebase_version"])
    table.add_row("Robot Type", meta["robot_type"])
    table.add_row("Total Episodes", str(meta["total_episodes"]))
    table.add_row("Total Frames", str(meta["total_frames"]))
    table.add_row("FPS", str(meta["fps"]))
    table.add_row("Avg Episode Length", f"{summary['avg_length']} frames")
    table.add_row("Avg Episode Duration", f"{summary['avg_duration']} s")
    table.add_row("Tasks", tasks[0]["task"])
    
    state_info = meta["features"]["observation.state"]

    table.add_row("State Dim", str(state_info["shape"][0]))
    table.add_row("Action Dim", str(meta["features"]["action"]["shape"][0]))
    table.add_row("Cameras", f"{len(cameras)} ({', '.join(cameras)})")
    table.add_row("State Joints", ", ".join(state_info["names"]))

    if json_output:
        import json
        result = {
            "dataset_path": os.path.abspath(dataset),
            "codebase_version": meta["codebase_version"],
            "robot_type": meta["robot_type"],
            "total_episodes": meta["total_episodes"],
            "total_frames": meta["total_frames"],
            "fps": meta["fps"],
            "avg_episode_length": summary["avg_length"],
            "avg_episode_duration": summary["avg_duration"],
            "tasks": [t["task"] for t in tasks],
            "state_dim": meta["features"]["observation.state"]["shape"][0],
            "action_dim": meta["features"]["action"]["shape"][0],
            "state_names": meta["features"]["observation.state"]["names"],
            "cameras": cameras,
        }
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(table)

@app.command()
def list_episodes(dataset: str=typer.Option(..., "--dataset", help="Path to the dataset directory.")):
    """List all episodes with frame counts and status."""

    episodes = list_episodes_summary(dataset)

    table = Table(title=f"[bold red]Episodes ({len(episodes)} total[/bold red])")
    table.add_column("Episode", style="cyan")
    table.add_column("Task")
    table.add_column("Frames")
    table.add_column("Duration")
    table.add_column("Videos")
    table.add_column("Status")

    for ep in episodes:
        # 给异常状态加颜色
        status_style = "green" if ep["status"] == "OK" else "yellow"
        table.add_row(
            str(ep["episode_index"]),
            str(ep["task"]),
            str(ep["frames"]),
            f"{ep['duration']}s",
            ep["videos"],
            f"[{status_style}]{ep['status']}[/{status_style}]",
        )

    console = Console()
    console.print(table)

@app.command()
def stats(
    dataset: str = "./dataset_0423_v2.1",
    output: str = typer.Option("reports/stats.json", "--output", "-o"),
    low_variance_threshold: float = typer.Option(0.0001, "--low-variance-threshold"),
):
    """Compute per-dimension statistics and generate figures."""
    from src.statistics import compute_stats, generate_figures, save_stats_json

    typer.echo("Computing statistics across all episodes...")
    result, action_matrix = compute_stats(dataset, low_variance_threshold)

    typer.echo(f"Saving stats to {output}")
    save_stats_json(result, output)

    figures_dir = os.path.join(os.path.dirname(output), "figures")
    typer.echo(f"Generating figures in {figures_dir}")
    saved = generate_figures(result, action_matrix, figures_dir)

    for p in saved:
        typer.echo(f"  Saved: {p}")

    typer.echo(f"\nState dims with low variance: {result['state']['_low_variance_dims']}")
    typer.echo(f"Action dims with low variance: {result['action']['_low_variance_dims']}")
    typer.echo(f"All-zero action rows: {result['action']['_all_zero_rows']}")
    typer.echo(f"NaN count: {result['state']['_nan_count']} (state), {result['action']['_nan_count']} (action)")
    typer.echo(f"Inf count: {result['state']['_inf_count']} (state), {result['action']['_inf_count']} (action)")

@app.command()
def check(
    dataset: str = "../dataset_0423_v2.1",
    output: str = typer.Option("reports/check_report.md", "--output", "-o"),
    min_frames: int = typer.Option(250, "--min-frames"),
    max_frames: int = typer.Option(350, "--max-frames"),
    max_action_jump: float = typer.Option(0.2, "--max-action-jump"),
    max_abs_action: float = typer.Option(2.0, "--max-abs-action"),
    timestamp_tolerance: float = typer.Option(0.005, "--timestamp-tolerance"),
    low_variance_threshold: float = typer.Option(0.0001, "--low-variance-threshold"),
):
    """Run data quality checks and export a Markdown report."""

    typer.echo("Running quality checks...\n")
    results = run_checks(
        dataset,
        min_frames=min_frames,
        max_frames=max_frames,
        max_action_jump=max_action_jump,
        max_abs_action=max_abs_action,
        timestamp_tolerance=timestamp_tolerance,
        low_variance_threshold=low_variance_threshold,
    )
    write_check_report(results, output)

    print_rich_summary(results)

@app.command()
def report(dataset: str = "../dataset_0423_v2.1", output: str = typer.Option("reports/dataset_report.md", "--output", "-o")):
    """Generate a full Markdown dataset analysis report."""

    typer.echo("Generating dataset report...")
    generate_report(dataset, output)
    typer.echo(f"Report saved to {output}")

@app.command()
def export(
    dataset: str = "../dataset_0423_v2.1",
    episodes: str = typer.Option(..., "--episodes", "-e", help="Comma-separated episode indices, e.g. 0,1,2"),
    output: str = typer.Option("exports/demo_small", "--output", "-o"),
):
    """Export a subset of episodes as a new dataset directory."""
    from src.exporter import export_episodes

    episode_list = [int(e.strip()) for e in episodes.split(",")]
    typer.echo(f"Exporting episodes {episode_list} → {output}")
    export_episodes(dataset, episode_list, output)

@app.command()
def replay(dataset: str = "../dataset_0423_v2.1", episode: int = typer.Option(0, "--episode", "-e")):
    """Replay a single episode frame-by-frame."""
    replay_episode(dataset, episode)

@app.command()
def web(
    dataset: str = typer.Option("../dataset_0423_v2.1", "--dataset", "-d"),
    share: bool = typer.Option(False, "--share", help="Create a public shareable link"),
):
    """Launch the Gradio web visualization interface."""
    from src.web_ui import launch_web

    launch_web(dataset, share=share)
    
if __name__ == "__main__":
    app()