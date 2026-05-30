from __future__ import annotations
from typing import Callable


def by_edd(blocks_data: list[dict]) -> list[int]:
    return sorted(
        range(len(blocks_data)),
        key=lambda i: (blocks_data[i]["due_date"], blocks_data[i]["processing_time"])
    )


def by_est(blocks_data: list[dict]) -> list[int]:
    return sorted(
        range(len(blocks_data)),
        key=lambda i: (blocks_data[i]["release_time"], blocks_data[i]["due_date"])
    )


def by_slack(blocks_data: list[dict]) -> list[int]:
    return sorted(
        range(len(blocks_data)),
        key=lambda i: (
            blocks_data[i]["due_date"] - blocks_data[i]["release_time"] - blocks_data[i]["processing_time"],
            blocks_data[i]["due_date"]
        )
    )


def by_spt(blocks_data: list[dict]) -> list[int]:
    return sorted(
        range(len(blocks_data)),
        key=lambda i: (blocks_data[i]["processing_time"], blocks_data[i]["due_date"])
    )


def by_weighted(blocks_data: list[dict]) -> list[int]:
    return sorted(
        range(len(blocks_data)),
        key=lambda i: (
            -(blocks_data[i]["due_date"] - blocks_data[i]["release_time"] - blocks_data[i]["processing_time"]),
            blocks_data[i]["processing_time"]
        )
    )


ALL_STRATEGIES: dict[str, Callable[[list[dict]], list[int]]] = {
    "edd": by_edd,
    "est": by_est,
    "slack": by_slack,
    "spt": by_spt,
    "weighted": by_weighted,
}
