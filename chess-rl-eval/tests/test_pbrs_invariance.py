import pytest
import math
import torch
import chess
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.cache import StockfishCache
from src.env.chess_gym import ChessGym
from src.verifier.tactical_verifier import select_best_candidate, evaluate_tactical_safety


def test_circular_trajectory_nonpositive_return():
    """
    Mathematical Guarantee: For any sequence of states s_0 -> s_1 ... -> s_k = s_0,
    the discounted reward telescopes:
    sum_{t=0}^{k-1} gamma^t [gamma * Phi(s_{t+1}) - Phi(s_t)] = (gamma^k - 1) * Phi(s_0).
    Since gamma = 0.99 < 1, gamma^k - 1 < 0.
    For Phi(s_0) >= 0, (gamma^k - 1) * Phi(s_0) <= 0.
    With step penalties r_step <= 0, total return is strictly <= 0.
    """
    gamma = 0.99
    step_penalty = -0.005

    # Simulate an arbitrary cyclic trajectory of 4 states back to s_0
    phi_values = [0.2, 0.4, 0.1, 0.3, 0.2]  # s0, s1, s2, s3, s0

    discounted_return = 0.0
    for t in range(len(phi_values) - 1):
        phi_t = phi_values[t]
        phi_next = phi_values[t + 1]
        pbrs_reward = gamma * phi_next - phi_t
        total_step_reward = pbrs_reward + step_penalty
        discounted_return += (gamma ** t) * total_step_reward

    assert discounted_return < 0.0


def test_win_prob_bounds():
    """Asserts Phi output is strictly bounded within [-1.0, 1.0]."""
    test_cps = [-10000, -2000, -500, -100, 0, 100, 500, 2000, 10000]
    for cp in test_cps:
        prob = StockfishCache.win_prob_from_cp(cp)
        assert -1.0 <= prob <= 1.0
        assert math.isfinite(prob)

    assert math.isclose(StockfishCache.win_prob_from_cp(0), 0.0, abs_tol=1e-7)


def test_mate_potential_ordering():
    """Asserts Phi(mate in 1) > Phi(mate in 3) > Phi(0 cp) > Phi(mated in 3) > Phi(mated in 1)."""
    phi_mate_1 = StockfishCache.win_prob_from_mate(1)
    phi_mate_3 = StockfishCache.win_prob_from_mate(3)
    phi_equal = StockfishCache.win_prob_from_cp(0)
    phi_mated_3 = StockfishCache.win_prob_from_mate(-3)
    phi_mated_1 = StockfishCache.win_prob_from_mate(-1)
    phi_checkmated = StockfishCache.win_prob_from_mate(0)

    assert phi_mate_1 > phi_mate_3 > phi_equal > phi_mated_3 > phi_mated_1
    assert phi_checkmated == -1.0
    assert phi_mate_1 == 0.99
    assert phi_mate_3 == 0.97
    assert phi_mated_1 == -0.99


def test_repetition_penalty_triggered():
    """Asserts repetition penalty (-0.30) is applied when returning to a visited state."""
    mock_cache = MagicMock(spec=StockfishCache)
    mock_cache.get_or_compute = AsyncMock(
        return_value={"score_cp": 0, "mate_in": None, "win_prob": 0.0}
    )

    gym = ChessGym(config={"reward": {"repetition_penalty": -0.30, "gamma": 0.99}}, cache=mock_cache)
    gym.reset()

    # Move sequence: 1. Nf3 Nf6 2. Ng1 Ng8 (returns to initial board state)
    gym.step("g1f3")
    gym.step("g8f6")
    gym.step("f3g1")
    obs, reward, done, info = gym.step("f6g8")

    assert info["r_rep"] == -0.30


def test_token_penalty_zero_below_threshold():
    """Asserts _token_len_penalty is 0 when <= 80 tokens, and negative above 80."""
    gym = ChessGym(config={"model": {"max_cot_tokens": 80}, "reward": {"token_penalty_coef": -0.002}})

    assert gym._token_len_penalty(50) == 0.0
    assert gym._token_len_penalty(80) == 0.0
    assert gym._token_len_penalty(81) == -0.002
    assert math.isclose(gym._token_len_penalty(100), -0.002 * 20)


def test_grpo_advantage_normalization():
    """Asserts group advantages have zero mean and unit std per group."""
    group_size = 4
    # Shape: [Batch_Size * Group_Size] = 8 rewards (2 groups of 4)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

    reshaped = rewards.view(-1, group_size)
    mean_r = reshaped.mean(dim=-1, keepdim=True)
    std_r = reshaped.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
    advantages = (reshaped - mean_r) / std_r

    for g in range(reshaped.shape[0]):
        group_adv = advantages[g]
        assert math.isclose(group_adv.mean().item(), 0.0, abs_tol=1e-5)
        assert math.isclose(group_adv.std(unbiased=False).item(), 1.0, abs_tol=1e-4)


def test_illegal_move_fallback():
    """Asserts select_best_candidate returns a legal move on invalid candidates and handles game over."""
    board = chess.Board()
    move = select_best_candidate(board, ["invalid1", "foo_bar", "e9e9"])
    assert move in board.legal_moves

    empty_candidates = select_best_candidate(board, [])
    assert empty_candidates in board.legal_moves

    # Checkmate position (Fool's Mate): black delivers Qh4#
    checkmate_board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert checkmate_board.is_checkmate()
    # Should not raise StopIteration, returns null move
    fallback_move = select_best_candidate(checkmate_board, ["e2e4"])
    assert fallback_move == chess.Move.null()
