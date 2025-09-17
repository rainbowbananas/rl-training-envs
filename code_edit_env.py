"""
CodeEditEnv: A custom reinforcement‑learning environment for code editing tasks.
------------------------------------------------------------------------------

This module implements a simple RL environment where an agent is tasked with
transforming an incorrect piece of source code into a target (correct) version
through a sequence of discrete edit operations. The environment is designed
to illustrate how one might build an RL environment that simulates code
editing actions such as replacing tokens or inserting lines. It follows the
Gymnasium API to ensure compatibility with standard RL libraries.

Key concepts and design choices:

* **Observation space** – The observation encodes the current source code as
  a character‑frequency vector. Although simplistic, this representation
  provides a fixed‑size numeric input suitable for deep learning models.
* **Action space** – A discrete set of edit operations. Each action is a
  tuple (pattern, replacement). When executed, the environment replaces the
  first occurrence of `pattern` in the code with `replacement`. A no‑op
  action leaves the code unchanged.
* **Reward function** – The reward is computed as the improvement in
  similarity between the current code and the target code after applying
  the chosen edit. We use the sequence matcher ratio from Python’s
  `difflib` as a proxy for similarity. Positive rewards indicate that the
  edit brought the code closer to the target.
* **Episode termination** – An episode ends when the code matches the
  target exactly or when a maximum number of edits (`max_steps`) has
  occurred.

This environment is intentionally simplified. In real applications you
might represent code using ASTs or token embeddings and apply more
sophisticated editing actions. You could also execute the code and use
test outcomes as the reward. Those extensions are left as exercises.

Author: OpenAI ChatGPT (Agent Mode)
Date: 2025‑09‑17
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Gymnasium is required. Install via pip install gymnasium"
    ) from exc


@dataclass
class CodeExample:
    """Container for a code editing task.

    Each example consists of an initial buggy snippet (`initial`), a
    ground‑truth correct version (`target`), and a list of allowed
    (pattern, replacement) edit operations. The environment will sample
    from this list when building an episode.
    """

    initial: str
    target: str
    edits: List[Tuple[str, str]]


class CodeEditEnv(gym.Env):
    """Gymnasium environment for iterative code editing.

    The agent receives a code vector observation and selects one of the
    available edits. The environment applies the edit, computes the reward
    based on similarity improvement, and returns the new observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        examples: List[CodeExample],
        max_steps: int = 5,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.examples = examples
        self.max_steps = max_steps
        self.random = random.Random(random_seed)
        # Build character vocabulary from all examples
        self.vocab = self._build_char_vocab(examples)
        self.obs_dim = len(self.vocab)
        # Action space size is derived from the maximum number of edits across examples
        self.max_actions = max(len(ex.edits) for ex in examples)
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Placeholder for current task
        self.current_example: Optional[CodeExample] = None
        self.current_code: str = ""
        self.steps_taken: int = 0

    def _build_char_vocab(self, examples: List[CodeExample]) -> Dict[str, int]:
        chars = set()
        for ex in examples:
            chars.update(ex.initial)
            chars.update(ex.target)
        return {ch: idx for idx, ch in enumerate(sorted(chars))}

    def _embed(self, code: str) -> np.ndarray:
        """Embed code as a normalized character frequency vector."""
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        for ch in code:
            if ch in self.vocab:
                vec[self.vocab[ch]] += 1.0
        if vec.max() > 0:
            vec /= vec.max()
        return vec

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # Sample a random code example
        self.current_example = self.random.choice(self.examples)
        self.current_code = self.current_example.initial
        self.steps_taken = 0
        obs = self._embed(self.current_code)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.current_example is not None, "reset() must be called before step()"
        # Clip action to valid range for current example
        valid_actions = len(self.current_example.edits)
        if action >= valid_actions:
            # Invalid actions are treated as no‑op with penalty
            chosen_edit = ("", "")
            reward = -0.1
        else:
            chosen_edit = self.current_example.edits[action]
            reward = 0.0
        before_sim = self._similarity(self.current_code, self.current_example.target)
        # Apply edit if applicable
        pattern, replacement = chosen_edit
        if pattern:
            self.current_code = self.current_code.replace(pattern, replacement, 1)
        after_sim = self._similarity(self.current_code, self.current_example.target)
        reward += after_sim - before_sim
        self.steps_taken += 1
        terminated = after_sim >= 1.0 or self.steps_taken >= self.max_steps
        observation = self._embed(self.current_code) if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return observation, reward, terminated, False, {}

    def _similarity(self, a: str, b: str) -> float:
        matcher = difflib.SequenceMatcher(None, a, b)
        return matcher.ratio()

    def render(self) -> None:  # pragma: no cover
        print(self.current_code)

    def close(self) -> None:  # pragma: no cover
        pass


def _demo_run() -> None:
    """Run a demo episode of CodeEditEnv using random actions."""
    examples = [
        CodeExample(
            initial="def add(a,b): return a-b",
            target="def add(a,b): return a+b",
            edits=[
                ("-", "+"),
                ("return a-b", "return a+b"),
                (",", ", "),  # formatting example
            ],
        ),
        CodeExample(
            initial="def multiply(x,y): prod=0\nfor i in range(y): prod+=x\nreturn prod",
            target="def multiply(x,y): return x*y",
            edits=[
                ("prod=0", ""),
                ("for i in range(y): prod+=x", ""),
                ("return prod", "return x*y"),
            ],
        ),
    ]
    env = CodeEditEnv(examples, max_steps=5, random_seed=42)
    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total += reward
    print("Demo episode reward:", total)


if __name__ == "__main__":  # pragma: no cover
    _demo_run()