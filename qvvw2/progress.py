from __future__ import annotations


def progress_hit(current: int, total: int, segments: int = 5) -> bool:
    """Print only at the first, last, and several intermediate milestones."""
    current = int(current)
    total = max(int(total), 1)
    segments = max(int(segments), 1)

    if current <= 1 or current >= total:
        return True

    stride = max(1, total // segments)
    return current % stride == 0


def progress_print(
    tag: str,
    current: int,
    total: int,
    detail: str = ""
) -> None:
    """Compact progress line suitable for Jupyter and terminal output."""
    current = int(current)
    total = max(int(total), 1)

    width = max(2, len(str(total)))
    pct = 100.0 * current / total

    msg = (
        f"[{tag}] "
        f"{current:{width}d}/{total:{width}d} "
        f"({pct:5.1f}%)"
    )

    if detail:
        msg += f" | {detail}"

    print(msg, flush=True)
