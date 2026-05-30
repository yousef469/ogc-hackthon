from __future__ import annotations
import math
import random
from typing import Optional

from utils import Bay, Block, check_entry, check_exit, check_collisions
from construction.helpers import block_bbox, time_overlaps


def try_swap_blocks(
    bid_a: int,
    bid_b: int,
    assignments: dict[int, dict],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
) -> bool:
    a_a = assignments[bid_a]
    a_b = assignments[bid_b]
    if a_a["bay_id"] != a_b["bay_id"]:
        return False

    bay_id = a_a["bay_id"]
    bay = bays[bay_id]

    blk_a = _find_block(bay_placed[bay_id], bid_a)
    blk_b = _find_block(bay_placed[bay_id], bid_b)
    if blk_a is None or blk_b is None:
        return False

    if not _can_place_at(bay, blk_a, a_b["x"], a_b["y"], bay_placed[bay_id]):
        return False
    if not _can_place_at(bay, blk_b, a_a["x"], a_a["y"], bay_placed[bay_id]):
        return False

    old_x_a, old_y_a = a_a["x"], a_a["y"]
    old_x_b, old_y_b = a_b["x"], a_b["y"]

    a_a["x"], a_a["y"] = a_b["x"], a_b["y"]
    a_b["x"], a_b["y"] = old_x_a, old_y_a
    blk_a.x, blk_a.y = a_a["x"], a_a["y"]
    blk_b.x, blk_b.y = a_b["x"], a_b["y"]

    return True


def try_move_block(
    bid: int,
    new_x: int,
    new_y: int,
    assignments: dict[int, dict],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
) -> bool:
    a = assignments[bid]
    bay_id = a["bay_id"]
    bay = bays[bay_id]

    blk = _find_block(bay_placed[bay_id], bid)
    if blk is None:
        return False

    if not _can_place_at(bay, blk, new_x, new_y, bay_placed[bay_id]):
        return False

    a["x"], a["y"] = new_x, new_y
    blk.x, blk.y = new_x, new_y
    return True


def try_rotate_block(
    bid: int,
    new_orient: int,
    assignments: dict[int, dict],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
) -> bool:
    a = assignments[bid]
    bay_id = a["bay_id"]
    bay = bays[bay_id]

    blk_data = blocks_data[bid]
    if new_orient >= len(blk_data["shape"]):
        return False

    bb = block_bbox(blk_data, new_orient)
    bw = bb[2] - bb[0]
    bh = bb[3] - bb[1]
    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
        return False

    old_orient = a["orient_idx"]
    a["orient_idx"] = new_orient

    new_blk = Block(
        block_id=bid, block_data=blk_data,
        x=a["x"], y=a["y"], orient_idx=new_orient,
    )

    if not bay.contains_block(new_blk):
        a["orient_idx"] = old_orient
        return False

    others = [b for b in bay_placed[bay_id] if b.block_id != bid]
    for ob in others:
        if check_collisions(bay, [new_blk, ob]):
            a["orient_idx"] = old_orient
            return False

    blk = _find_block(bay_placed[bay_id], bid)
    if blk is not None:
        blk.orient_idx = new_orient
        blk.block_data = blk_data

    return True


def try_time_shift(
    bid: int,
    new_entry: int,
    assignments: dict[int, dict],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
) -> bool:
    a = assignments[bid]
    blk_data = blocks_data[bid]
    r_time = blk_data["release_time"]
    proc = blk_data["processing_time"]

    if new_entry < r_time:
        return False

    new_exit = new_entry + proc
    bay_id = a["bay_id"]
    bay = bays[bay_id]

    blk = Block(
        block_id=bid, block_data=blk_data,
        x=a["x"], y=a["y"], orient_idx=a["orient_idx"],
    )

    others_in_bay = [
        b for b in bay_placed[bay_id] if b.block_id != bid
    ]
    other_sched = [
        s for i, s in enumerate(bay_schedule[bay_id])
        if bay_placed[bay_id][i].block_id != bid
    ]

    present_at_entry = [
        b for b, (e1, e2) in zip(others_in_bay, other_sched)
        if e1 <= new_entry < e2
    ]
    if check_entry(bay, present_at_entry, blk, fast=True):
        return False

    present_at_exit = [blk] + [
        b for b, (e1, e2) in zip(others_in_bay, other_sched)
        if e1 < new_exit < e2
    ]
    if check_exit(bay, present_at_exit, blk, fast=True):
        return False

    for b_other, (e1, e2) in zip(others_in_bay, other_sched):
        if e1 <= new_entry or e2 >= new_exit:
            continue
        if not time_overlaps(new_entry, new_exit, e1, e2):
            continue
        if check_collisions(bay, [blk, b_other]):
            return False

    for i, s in enumerate(bay_schedule[bay_id]):
        if bay_placed[bay_id][i].block_id == bid:
            bay_schedule[bay_id][i] = (new_entry, new_exit)
            break

    a["entry_time"] = new_entry
    a["exit_time"] = new_exit
    return True


def try_reassign_bay(
    bid: int,
    new_bay: int,
    new_x: int,
    new_y: int,
    new_orient: int | None,
    new_entry: int | None,
    assignments: dict[int, dict],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
) -> bool:
    a = assignments[bid]
    blk_data = blocks_data[bid]
    old_bay = a["bay_id"]
    r_time = blk_data["release_time"]
    proc = blk_data["processing_time"]

    if new_orient is None:
        new_orient = a["orient_idx"]
    if new_entry is None:
        new_entry = a["entry_time"]

    new_exit = new_entry + proc
    if new_entry < r_time:
        return False

    bb = block_bbox(blk_data, new_orient)
    bw = bb[2] - bb[0]
    bh = bb[3] - bb[1]
    if bw > bays[new_bay].width + 1e-6 or bh > bays[new_bay].height + 1e-6:
        return False

    new_blk = Block(
        block_id=bid, block_data=blk_data,
        x=new_x, y=new_y, orient_idx=new_orient,
    )
    if not bays[new_bay].contains_block(new_blk):
        return False

    others_new = [
        b for b in bay_placed[new_bay]
    ]
    other_sched_new = list(bay_schedule[new_bay])

    present_at_entry = [
        b for b, (e1, e2) in zip(others_new, other_sched_new)
        if e1 <= new_entry < e2
    ]
    if check_entry(bays[new_bay], present_at_entry, new_blk, fast=True):
        return False

    present_at_exit = [new_blk] + [
        b for b, (e1, e2) in zip(others_new, other_sched_new)
        if e1 < new_exit < e2
    ]
    if check_exit(bays[new_bay], present_at_exit, new_blk, fast=True):
        return False

    for b_other, (e1, e2) in zip(others_new, other_sched_new):
        if e1 <= new_entry or e2 >= new_exit:
            continue
        if not time_overlaps(new_entry, new_exit, e1, e2):
            continue
        if check_collisions(bays[new_bay], [new_blk, b_other]):
            return False

    for i, b in enumerate(bay_placed[old_bay]):
        if b.block_id == bid:
            bay_placed[old_bay].pop(i)
            bay_schedule[old_bay].pop(i)
            break
    bay_loads[old_bay] -= blk_data["workload"]

    bay_placed[new_bay].append(new_blk)
    bay_schedule[new_bay].append((new_entry, new_exit))
    bay_loads[new_bay] += blk_data["workload"]

    a["bay_id"] = new_bay
    a["x"] = new_x
    a["y"] = new_y
    a["orient_idx"] = new_orient
    a["entry_time"] = new_entry
    a["exit_time"] = new_exit
    return True


def _find_block(blocks: list[Block], bid: int) -> Optional[Block]:
    for b in blocks:
        if b.block_id == bid:
            return b
    return None


def _can_place_at(bay: Bay, blk: Block, x: int, y: int,
                  all_blocks: list[Block]) -> bool:
    old_x, old_y = blk.x, blk.y
    blk.x, blk.y = x, y

    if not bay.contains_block(blk):
        blk.x, blk.y = old_x, old_y
        return False

    others = [b for b in all_blocks if b.block_id != blk.block_id]
    for ob in others:
        if check_collisions(bay, [blk, ob]):
            blk.x, blk.y = old_x, old_y
            return False

    blk.x, blk.y = old_x, old_y
    return True
