#!/usr/bin/env python3
"""
Utilities for recursive search of allele sets that explain TCR occurrence patterns.

Key practical safeguards vs. naive Alg.1:
- Do NOT branch on alleles that explain zero remaining positive donors.
- Prefer enrichment-only candidates (avoid chi-square "depletion" branches).
- Shrink the allele universe to those seen among positive donors.
- Offer non-DFS search modes (beam/BFS-ish and Monte-Carlo rollouts) to budget compute
  with less truncation bias than DFS cutoffs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Tuple, Literal

import numpy as np

# Chi-squared(1) ~= 3.84 at p=0.05
DEFAULT_MIN_CHI_SQUARED = 3.841

SearchMode = Literal["dfs", "beam", "mc"]
BeamScore = Literal["remaining", "chi_sum", "hybrid"]


def _state_key(y: np.ndarray, cand: np.ndarray) -> tuple[int, bytes, int, bytes]:
    y_bits = bytes(np.packbits(np.asarray(y, dtype=np.uint8)))
    cand_bits = bytes(np.packbits(np.asarray(cand, dtype=np.uint8)))
    return (int(y.size), y_bits, int(cand.size), cand_bits)


def rank_alleles(
    y: np.ndarray,
    x_prime: np.ndarray,
    cand: np.ndarray,
    counts_a: np.ndarray,
    counts_total: int,
    min_chi_squared: Optional[float] = DEFAULT_MIN_CHI_SQUARED,
    *,
    require_coverage: bool = True,
    enrichment_only: bool = True,
    return_scores: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Rank candidate alleles by (enrichment-only) chi-squared among remaining samples.

    Practical changes vs. naive Alg.1:
      - require_coverage: only consider alleles that cover >=1 remaining positive donor.
      - enrichment_only: only consider alleles with counts_ia > expected (avoid depletion branches).

    Returns:
      allele indices sorted by descending chi-squared, filtered by threshold.
      If return_scores=True, returns (indices, chi_scores_for_indices).
    """
    if counts_total <= 0:
        if return_scores:
            return np.array([], dtype=int), np.array([], dtype=np.float64)
        return np.array([], dtype=int)

    y_vec = np.asarray(y, dtype=np.float64)
    x_mat = np.asarray(x_prime)
    cand_vec = np.asarray(cand, dtype=np.float64)
    counts_a_vec = np.asarray(counts_a, dtype=np.float64)

    counts_i = float(np.sum(y_vec))
    if counts_i <= 0:
        if return_scores:
            return np.array([], dtype=int), np.array([], dtype=np.float64)
        return np.array([], dtype=int)

    # counts among remaining unexplained positives
    counts_ia = y_vec @ x_mat  # shape (A,)
    counts_exp_ia = counts_i * counts_a_vec / float(counts_total)

    # Eligibility mask
    eligible = (cand_vec != 0) & (counts_exp_ia > 0)
    if require_coverage:
        eligible &= (counts_ia > 0)
    diff = counts_ia - counts_exp_ia
    if enrichment_only:
        eligible &= (diff > 0)

    chi = np.zeros_like(counts_exp_ia, dtype=np.float64)
    if np.any(eligible):
        d = diff[eligible]
        chi[eligible] = (d * d) / counts_exp_ia[eligible]

    # Thresholding
    if min_chi_squared is not None:
        eligible &= (chi > min_chi_squared)

    if not np.any(eligible):
        if return_scores:
            return np.array([], dtype=int), np.array([], dtype=np.float64)
        return np.array([], dtype=int)

    idx = np.flatnonzero(eligible)
    order = idx[np.argsort(-chi[idx])]
    if return_scores:
        return order, chi[order]
    return order


def _update_stats_timeout(stats: dict) -> None:
    stats["truncated"] = True
    stats["timeout_hit"] = True


def _update_stats_max_solutions(stats: dict) -> None:
    stats["truncated"] = True
    stats["max_solutions_hit"] = True


def _update_stats_max_depth(stats: dict) -> None:
    stats["truncated"] = True
    stats["max_depth_hit"] = True


def _update_stats_max_branch(stats: dict) -> None:
    stats["truncated"] = True
    stats["max_branch_capped"] = True


def _update_stats_no_candidate(stats: dict) -> None:
    stats["no_candidate_hit"] = True


def _select_next_allele_dfs(
    z: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    cand: np.ndarray,
    counts_a: np.ndarray,
    counts_total: int,
    min_chi_squared: Optional[float],
    *,
    max_solutions: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_branch: Optional[int] = None,
    max_runtime_sec: Optional[float] = None,
    _depth: int = 0,
    _start_time: Optional[float] = None,
    _stats: Optional[dict] = None,
    _memo: Optional[set] = None,
) -> List[np.ndarray]:
    """
    DFS recursion with hard safeguards:
      - skip alleles that don't reduce remaining uncovered positives (no-progress guard).
      - rank_alleles requires coverage and (by default) enrichment-only.
    """
    solutions: List[np.ndarray] = []
    if _start_time is None:
        _start_time = time.perf_counter()
    if _stats is None:
        _stats = {
            "solutions": 0,
            "truncated": False,
            "max_solutions_hit": False,
            "max_depth_hit": False,
            "max_branch_capped": False,
            "timeout_hit": False,
            "no_candidate_hit": False,
        }

    # budgets
    if max_runtime_sec is not None and (time.perf_counter() - _start_time > max_runtime_sec):
        _update_stats_timeout(_stats)
        return solutions
    if max_solutions is not None and _stats["solutions"] >= max_solutions:
        _update_stats_max_solutions(_stats)
        return solutions
    if _memo is not None:
        key = _state_key(y, cand)
        if key in _memo:
            return solutions
        _memo.add(key)

    ranked = rank_alleles(
        y, x_prime, cand, counts_a, counts_total,
        min_chi_squared=min_chi_squared,
        require_coverage=True,
        enrichment_only=True,
    )
    if ranked.size == 0:
        if int(np.sum(y)) > 0:
            _update_stats_no_candidate(_stats)
        return solutions
    if max_branch is not None and ranked.size > max_branch:
        ranked = ranked[:max_branch]
        _update_stats_max_branch(_stats)

    y_sum = int(np.sum(y))

    for a in ranked:
        if max_runtime_sec is not None and (time.perf_counter() - _start_time > max_runtime_sec):
            _update_stats_timeout(_stats)
            break
        if max_solutions is not None and _stats["solutions"] >= max_solutions:
            _update_stats_max_solutions(_stats)
            break

        z_next = z.copy()
        z_next[a] = 1

        y_next = y.copy()
        y_next[x_prime[:, a] == 1] = 0

        # No-progress guard (should be redundant with require_coverage, but keep as a hard safety)
        if int(np.sum(y_next)) == y_sum:
            continue

        cand_next = cand.copy()
        cand_next[a] = 0

        if np.sum(y_next) == 0:
            solutions.append(z_next)
            _stats["solutions"] += 1
            continue

        if max_depth is not None and (_depth + 1) >= max_depth:
            _update_stats_max_depth(_stats)
            continue

        solutions.extend(
            _select_next_allele_dfs(
                z_next,
                y_next,
                x_prime,
                cand_next,
                counts_a,
                counts_total,
                min_chi_squared,
                max_solutions=max_solutions,
                max_depth=max_depth,
                max_branch=max_branch,
                max_runtime_sec=max_runtime_sec,
                _depth=_depth + 1,
                _start_time=_start_time,
                _stats=_stats,
            )
        )
    return solutions


@dataclass(frozen=True)
class _BeamState:
    z: np.ndarray
    y: np.ndarray
    cand: np.ndarray
    depth: int
    score: float


def _select_next_allele_beam(
    z0: np.ndarray,
    y0: np.ndarray,
    x_prime: np.ndarray,
    cand0: np.ndarray,
    counts_a: np.ndarray,
    counts_total: int,
    min_chi_squared: Optional[float],
    *,
    beam_width: int = 256,
    beam_score: BeamScore = "chi_sum",
    max_solutions: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_branch: Optional[int] = None,
    max_runtime_sec: Optional[float] = None,
    use_dp: bool = True,
    _start_time: Optional[float] = None,
    _stats: Optional[dict] = None,
) -> List[np.ndarray]:
    """
    BFS-ish beam search:
      - expand frontier level-by-level (reduces DFS truncation bias)
      - keep at most beam_width partial states (by smallest remaining uncovered)
      - still uses chi-square only as a proposer, not as a posterior scorer
    """
    solutions: List[np.ndarray] = []
    if _start_time is None:
        _start_time = time.perf_counter()
    if _stats is None:
        _stats = {
            "solutions": 0,
            "truncated": False,
            "max_solutions_hit": False,
            "max_depth_hit": False,
            "max_branch_capped": False,
            "timeout_hit": False,
            "no_candidate_hit": False,
        }

    frontier: List[_BeamState] = [_BeamState(z=z0, y=y0, cand=cand0, depth=0, score=0.0)]

    # If no max_depth is set, allow full depth up to remaining positives
    if max_depth is None:
        max_depth = int(np.sum(y0))

    for depth in range(max_depth):
        if max_runtime_sec is not None and (time.perf_counter() - _start_time > max_runtime_sec):
            _update_stats_timeout(_stats)
            break
        if max_solutions is not None and _stats["solutions"] >= max_solutions:
            _update_stats_max_solutions(_stats)
            break

        next_states: List[Tuple[int, float, _BeamState]] = []
        # score = remaining uncovered (smaller is better)
        for st in frontier:
            if max_runtime_sec is not None and (time.perf_counter() - _start_time > max_runtime_sec):
                _update_stats_timeout(_stats)
                break
            if max_solutions is not None and _stats["solutions"] >= max_solutions:
                _update_stats_max_solutions(_stats)
                break

            if beam_score in ("chi_sum", "hybrid"):
                ranked, chi = rank_alleles(
                    st.y, x_prime, st.cand, counts_a, counts_total,
                    min_chi_squared=min_chi_squared,
                    require_coverage=True,
                    enrichment_only=True,
                    return_scores=True,
                )
            else:
                ranked = rank_alleles(
                    st.y, x_prime, st.cand, counts_a, counts_total,
                    min_chi_squared=min_chi_squared,
                    require_coverage=True,
                    enrichment_only=True,
                )
            if ranked.size == 0:
                if int(np.sum(st.y)) > 0:
                    _update_stats_no_candidate(_stats)
                continue
            if max_branch is not None and ranked.size > max_branch:
                ranked = ranked[:max_branch]
                if beam_score in ("chi_sum", "hybrid"):
                    chi = chi[:max_branch]
                _update_stats_max_branch(_stats)

            y_sum = int(np.sum(st.y))
            if beam_score in ("chi_sum", "hybrid"):
                iter_pairs = zip(ranked, chi)
            else:
                iter_pairs = ((a, 0.0) for a in ranked)
            for a, chi_val in iter_pairs:
                z_next = st.z.copy()
                z_next[a] = 1
                y_next = st.y.copy()
                y_next[x_prime[:, a] == 1] = 0

                # No-progress guard
                rem = int(np.sum(y_next))
                if rem == y_sum:
                    continue

                cand_next = st.cand.copy()
                cand_next[a] = 0

                if rem == 0:
                    solutions.append(z_next)
                    _stats["solutions"] += 1
                    if max_solutions is not None and _stats["solutions"] >= max_solutions:
                        _update_stats_max_solutions(_stats)
                        break
                else:
                    next_score = st.score + float(chi_val)
                    next_states.append(
                        (rem, next_score, _BeamState(
                            z=z_next,
                            y=y_next,
                            cand=cand_next,
                            depth=st.depth + 1,
                            score=next_score,
                        ))
                    )

            if max_solutions is not None and _stats["solutions"] >= max_solutions:
                break

        if not next_states:
            break
        if use_dp:
            merged: dict[tuple[int, bytes, int, bytes], tuple[int, float, _BeamState]] = {}
            for rem, score, st in next_states:
                key = _state_key(st.y, st.cand)
                if key not in merged:
                    merged[key] = (rem, score, st)
                else:
                    prev_rem, prev_score, _ = merged[key]
                    if beam_score == "remaining":
                        if rem < prev_rem:
                            merged[key] = (rem, score, st)
                    elif beam_score == "chi_sum":
                        if score > prev_score:
                            merged[key] = (rem, score, st)
                    else:
                        if (rem < prev_rem) or (rem == prev_rem and score > prev_score):
                            merged[key] = (rem, score, st)
            next_states = list(merged.values())

        # Beam prune: pick by score policy
        if beam_score == "remaining":
            next_states.sort(key=lambda t: t[0])
        elif beam_score == "chi_sum":
            next_states.sort(key=lambda t: (-t[1], t[0]))
        else:
            next_states.sort(key=lambda t: (t[0], -t[1]))

        if len(next_states) > beam_width:
            next_states = next_states[:beam_width]
            _stats["truncated"] = True

        frontier = [st for _, __, st in next_states]

    return solutions


def _select_next_allele_monte_carlo(
    z0: np.ndarray,
    y0: np.ndarray,
    x_prime: np.ndarray,
    cand0: np.ndarray,
    counts_a: np.ndarray,
    counts_total: int,
    min_chi_squared: Optional[float],
    *,
    num_rollouts: int = 1024,
    top_k: int = 20,
    temperature: float = 1.0,
    max_depth: int = 8,
    max_solutions: Optional[int] = None,
    max_runtime_sec: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    _start_time: Optional[float] = None,
    _stats: Optional[dict] = None,
) -> List[np.ndarray]:
    """
    Monte-Carlo rollouts:
      - each rollout samples a branch (stochastic) from the top candidates
      - yields diverse solutions under a fixed compute budget
    """
    solutions: List[np.ndarray] = []
    if rng is None:
        rng = np.random.default_rng()

    if _start_time is None:
        _start_time = time.perf_counter()
    if _stats is None:
        _stats = {
            "solutions": 0,
            "truncated": False,
            "max_solutions_hit": False,
            "max_depth_hit": False,
            "max_branch_capped": False,
            "timeout_hit": False,
            "no_candidate_hit": False,
        }

    # de-duplicate solutions by packed bits
    seen = set()

    for _ in range(num_rollouts):
        if max_runtime_sec is not None and (time.perf_counter() - _start_time > max_runtime_sec):
            _update_stats_timeout(_stats)
            break
        if max_solutions is not None and _stats["solutions"] >= max_solutions:
            _update_stats_max_solutions(_stats)
            break

        z = z0.copy()
        y = y0.copy()
        cand = cand0.copy()

        for depth in range(max_depth):
            if np.sum(y) == 0:
                break

            ranked, chi = rank_alleles(
                y, x_prime, cand, counts_a, counts_total,
                min_chi_squared=min_chi_squared,
                require_coverage=True,
                enrichment_only=True,
                return_scores=True,
            )
            if ranked.size == 0:
                if int(np.sum(y)) > 0:
                    _update_stats_no_candidate(_stats)
                break

            # sample from top_k using softmax over chi/temperature
            kk = min(top_k, ranked.size)
            r = ranked[:kk]
            s = chi[:kk].astype(np.float64)

            if temperature <= 0:
                a = r[0]
            else:
                logits = s / float(temperature)
                logits -= np.max(logits)
                p = np.exp(logits)
                p /= np.sum(p)
                a = rng.choice(r, p=p)

            y_sum = int(np.sum(y))
            z[a] = 1
            y[x_prime[:, a] == 1] = 0
            cand[a] = 0

            # no-progress (extra safety)
            if int(np.sum(y)) == y_sum:
                break

        if np.sum(y) == 0:
            key = bytes(np.packbits(z.astype(np.uint8)))
            if key not in seen:
                seen.add(key)
                solutions.append(z.copy())
                _stats["solutions"] += 1

    return solutions


def _shrink_to_positive_alleles(
    x_mat: np.ndarray,
    mask: np.ndarray,
    counts_a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict allele columns to those present in at least one positive donor.
    Returns: (x_prime_shrunk, counts_a_shrunk, index_map)
    """
    x_prime = x_mat[mask, :]
    col_mask = np.sum(x_prime, axis=0) > 0
    idx_map = np.flatnonzero(col_mask)
    return x_prime[:, col_mask], counts_a[col_mask], idx_map


def _expand_solution(z_sub: np.ndarray, a_total: int, idx_map: np.ndarray) -> np.ndarray:
    z_full = np.zeros(a_total, dtype=np.uint8)
    z_full[idx_map] = z_sub.astype(np.uint8, copy=False)
    return z_full


def find_solutions(
    y_i: Union[Sequence[int], np.ndarray],
    x: np.ndarray,
    min_chi_squared: Optional[float] = DEFAULT_MIN_CHI_SQUARED,
    *,
    search_mode: SearchMode = "beam",
    beam_width: int = 256,
    beam_score: BeamScore = "chi_sum",
    mc_num_rollouts: int = 1024,
    mc_top_k: int = 20,
    mc_temperature: float = 1.0,
    max_solutions: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_branch: Optional[int] = None,
    max_runtime_sec: Optional[float] = None,
    use_dp: bool = True,
    return_stats: bool = False,
) -> List[np.ndarray] | tuple[List[np.ndarray], dict]:
    """
    Find candidate allele sets that explain all samples where y_i == 1.

    Practical defaults:
      - search_mode="beam" to avoid DFS truncation bias under budgets.
      - enrichment-only + coverage-only candidate proposals.
      - shrink allele columns to those seen in positives.

    Args:
        y_i: Binary vector (length N) indicating TCR presence across samples.
        x: Allele presence/absence matrix (shape N x A).
        min_chi_squared: Threshold for candidate allele ranking; set to None to disable.
        search_mode: "dfs", "beam", or "mc".
        beam_width: Beam size for "beam".
        mc_*: Monte-Carlo parameters for "mc".
        max_solutions: Optional cap on number of solutions to return.
        max_depth: Optional cap on recursion depth (max alleles in a set).
        max_branch: Optional cap on candidate alleles per recursion step.
        max_runtime_sec: Optional wall-time cap for the search.
        return_stats: If True, return (solutions, stats) with truncation flags.
    """
    y_i_vec = np.asarray(y_i)
    x_mat = np.asarray(x)
    if y_i_vec.ndim != 1:
        raise ValueError("y_i must be a 1D vector of length N.")
    if x_mat.ndim != 2:
        raise ValueError("x must be a 2D array with shape N x A.")
    if x_mat.shape[0] != y_i_vec.shape[0]:
        raise ValueError("x and y_i must have matching N dimension.")

    mask = y_i_vec != 0
    a_total = x_mat.shape[1]
    if not np.any(mask):
        out = [np.zeros(a_total, dtype=np.uint8)]
        if return_stats:
            return out, {
                "solutions": 1,
                "truncated": False,
                "max_solutions_hit": False,
                "max_depth_hit": False,
                "max_branch_capped": False,
                "timeout_hit": False,
                "no_candidate_hit": False,
            }
        return out

    counts_a_full = np.sum(x_mat, axis=0)
    x_prime, counts_a, idx_map = _shrink_to_positive_alleles(x_mat, mask, counts_a_full)

    # If after shrinking there are no alleles at all (should be rare), return empty solution
    if x_prime.shape[1] == 0:
        out = [np.zeros(a_total, dtype=np.uint8)]
        if return_stats:
            return out, {
                "solutions": 1,
                "truncated": True,
                "max_solutions_hit": False,
                "max_depth_hit": False,
                "max_branch_capped": False,
                "timeout_hit": False,
                "no_candidate_hit": True,
            }
        return out

    y = np.ones(x_prime.shape[0], dtype=np.uint8)
    z0 = np.zeros(x_prime.shape[1], dtype=np.uint8)
    cand0 = np.ones(x_prime.shape[1], dtype=np.uint8)

    stats = {
        "solutions": 0,
        "truncated": False,
        "max_solutions_hit": False,
        "max_depth_hit": False,
        "max_branch_capped": False,
        "timeout_hit": False,
        "no_candidate_hit": False,
    }

    start = time.perf_counter()
    if search_mode == "dfs":
        memo = set() if use_dp else None
        sub_solutions = _select_next_allele_dfs(
            z0, y, x_prime, cand0, counts_a, x_mat.shape[0], min_chi_squared,
            max_solutions=max_solutions,
            max_depth=max_depth,
            max_branch=max_branch,
            max_runtime_sec=max_runtime_sec,
            _start_time=start,
            _stats=stats,
            _memo=memo,
        )
    elif search_mode == "beam":
        sub_solutions = _select_next_allele_beam(
            z0, y, x_prime, cand0, counts_a, x_mat.shape[0], min_chi_squared,
            beam_width=beam_width,
            beam_score=beam_score,
            max_solutions=max_solutions,
            max_depth=max_depth,
            max_branch=max_branch,
            max_runtime_sec=max_runtime_sec,
            use_dp=use_dp,
            _start_time=start,
            _stats=stats,
        )
    elif search_mode == "mc":
        # If no max_depth is set, allow full depth up to remaining positives
        md = max_depth if max_depth is not None else int(y.shape[0])
        sub_solutions = _select_next_allele_monte_carlo(
            z0, y, x_prime, cand0, counts_a, x_mat.shape[0], min_chi_squared,
            num_rollouts=mc_num_rollouts,
            top_k=mc_top_k,
            temperature=mc_temperature,
            max_depth=md,
            max_solutions=max_solutions,
            max_runtime_sec=max_runtime_sec,
            _start_time=start,
            _stats=stats,
        )
    else:
        raise ValueError(f"Unknown search_mode={search_mode!r}")

    # Expand back to full A-length solutions
    solutions = [_expand_solution(z_sub, a_total=a_total, idx_map=idx_map) for z_sub in sub_solutions]

    if return_stats:
        return solutions, stats
    return solutions


def find_solutions_from_indices(
    donor_indices: Union[Sequence[int], np.ndarray],
    x: np.ndarray,
    min_chi_squared: Optional[float] = DEFAULT_MIN_CHI_SQUARED,
    *,
    counts_a: Optional[np.ndarray] = None,
    counts_total: Optional[int] = None,
    search_mode: SearchMode = "beam",
    beam_width: int = 256,
    beam_score: BeamScore = "chi_sum",
    mc_num_rollouts: int = 1024,
    mc_top_k: int = 20,
    mc_temperature: float = 1.0,
    max_solutions: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_branch: Optional[int] = None,
    max_runtime_sec: Optional[float] = None,
    use_dp: bool = True,
    return_stats: bool = False,
) -> List[np.ndarray] | tuple[List[np.ndarray], dict]:
    """
    Find candidate allele sets using donor indices (avoids building a full y_i vector).
    """
    x_mat = np.asarray(x)
    idx = np.asarray(donor_indices, dtype=int)
    a_total = x_mat.shape[1]
    if idx.ndim != 1:
        raise ValueError("donor_indices must be a 1D sequence of indices.")
    if idx.size == 0:
        out = [np.zeros(a_total, dtype=np.uint8)]
        if return_stats:
            return out, {
                "solutions": 1,
                "truncated": False,
                "max_solutions_hit": False,
                "max_depth_hit": False,
                "max_branch_capped": False,
                "timeout_hit": False,
                "no_candidate_hit": False,
            }
        return out

    if counts_a is None:
        counts_a = np.sum(x_mat, axis=0)
    if counts_total is None:
        counts_total = x_mat.shape[0]

    # shrink allele columns to those present among positive donors for this TCR
    x_prime = x_mat[idx, :]
    col_mask = np.sum(x_prime, axis=0) > 0
    idx_map = np.flatnonzero(col_mask)
    x_prime = x_prime[:, col_mask]
    counts_a_sub = np.asarray(counts_a)[col_mask]

    if x_prime.shape[1] == 0:
        out = [np.zeros(a_total, dtype=np.uint8)]
        if return_stats:
            return out, {
                "solutions": 1,
                "truncated": True,
                "max_solutions_hit": False,
                "max_depth_hit": False,
                "max_branch_capped": False,
                "timeout_hit": False,
                "no_candidate_hit": True,
            }
        return out

    y = np.ones(x_prime.shape[0], dtype=np.uint8)
    z0 = np.zeros(x_prime.shape[1], dtype=np.uint8)
    cand0 = np.ones(x_prime.shape[1], dtype=np.uint8)

    stats = {
        "solutions": 0,
        "truncated": False,
        "max_solutions_hit": False,
        "max_depth_hit": False,
        "max_branch_capped": False,
        "timeout_hit": False,
        "no_candidate_hit": False,
    }

    start = time.perf_counter()
    if search_mode == "dfs":
        memo = set() if use_dp else None
        sub_solutions = _select_next_allele_dfs(
            z0, y, x_prime, cand0, counts_a_sub, counts_total, min_chi_squared,
            max_solutions=max_solutions,
            max_depth=max_depth,
            max_branch=max_branch,
            max_runtime_sec=max_runtime_sec,
            _start_time=start,
            _stats=stats,
            _memo=memo,
        )
    elif search_mode == "beam":
        sub_solutions = _select_next_allele_beam(
            z0, y, x_prime, cand0, counts_a_sub, counts_total, min_chi_squared,
            beam_width=beam_width,
            beam_score=beam_score,
            max_solutions=max_solutions,
            max_depth=max_depth,
            max_branch=max_branch,
            max_runtime_sec=max_runtime_sec,
            use_dp=use_dp,
            _start_time=start,
            _stats=stats,
        )
    elif search_mode == "mc":
        md = max_depth if max_depth is not None else int(y.shape[0])
        sub_solutions = _select_next_allele_monte_carlo(
            z0, y, x_prime, cand0, counts_a_sub, counts_total, min_chi_squared,
            num_rollouts=mc_num_rollouts,
            top_k=mc_top_k,
            temperature=mc_temperature,
            max_depth=md,
            max_solutions=max_solutions,
            max_runtime_sec=max_runtime_sec,
            _start_time=start,
            _stats=stats,
        )
    else:
        raise ValueError(f"Unknown search_mode={search_mode!r}")

    solutions = [_expand_solution(z_sub, a_total=a_total, idx_map=idx_map) for z_sub in sub_solutions]

    if return_stats:
        return solutions, stats
    return solutions
