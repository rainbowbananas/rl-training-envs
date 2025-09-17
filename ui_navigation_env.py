"""
FormNavigationEnv: A reinforcement‑learning environment for form‑filling tasks.
--------------------------------------------------------------------------

This module defines a simple UI navigation environment where an agent must
complete a form by filling in the correct values and submitting. It is a
high‑level simulation of interacting with UI elements such as text fields
and buttons. Rather than driving a real browser, the environment keeps
track of form state internally. It can serve as the starting point for
wrapping real browser automation frameworks like Selenium or Playwright.

Key features:

* **Observation space** – A binary vector indicating which fields have been
  filled correctly. This keeps the observation minimal and focuses the
  agent on the relevant task status.
* **Action space** – Each field has one action that fills it correctly.
  An additional action triggers the submit button. Invalid actions are
  penalized (e.g., trying to fill an already filled field).
* **Reward function** – The agent receives a small negative penalty for
  each step to encourage efficiency. A larger positive reward is given
  when the form is submitted with all fields correct. Submitting early
  yields zero reward and ends the episode.
* **Episode termination** – An episode ends when the submit action is
  taken or after a maximum number of steps.

This simplified environment encourages the agent to fill all required
fields then submit. To extend this for real UI automation, replace the
internal state updates with calls to browser automation libraries.

Author: OpenAI ChatGPT (Agent Mode)
Date: 2025‑09‑17
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Gymnasium is required. Install via pip install gymnasium"
    ) from exc


@dataclass
class FormDefinition:
    """Defines a form with required fields and their correct values."""

    fields: Dict[str, str]


class FormNavigationEnv(gym.Env):
    """Environment simulating navigation and completion of a form."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        forms: List[FormDefinition],
        max_steps: int = 10,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.forms = forms
        self.max_steps = max_steps
        self.random = random.Random(random_seed)
        # Number of fields is determined by the largest form
        self.max_fields = max(len(form.fields) for form in forms)
        # Actions: one per field + 1 submit action
        self.action_space = spaces.Discrete(self.max_fields + 1)
        # Observation: binary vector for each field (1 if filled correctly)
        self.observation_space = spaces.MultiBinary(self.max_fields)
        self.current_form: Optional[FormDefinition] = None
        self.field_status: List[int] = []
        self.steps_taken: int = 0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_form = self.random.choice(self.forms)
        num_fields = len(self.current_form.fields)
        # Initialize all fields as unfilled (0). Extra positions are padded.
        self.field_status = [0] * self.max_fields
        self.steps_taken = 0
        return np.array(self.field_status, dtype=np.int8), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.current_form is not None, "reset() must be called before step()"
        reward = -0.01  # small step penalty
        done = False
        num_fields = len(self.current_form.fields)
        if action < num_fields:
            # Fill field action
            if self.field_status[action] == 1:
                # Field already filled; penalize lightly
                reward -= 0.05
            else:
                # Mark as filled correctly
                self.field_status[action] = 1
                reward += 0.1
        else:
            # Submit action
            if all(self.field_status[i] == 1 for i in range(num_fields)):
                reward += 1.0  # success reward
            # Episode ends on submit regardless of success
            done = True
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True
        observation = np.array(self.field_status, dtype=np.int8) if not done else np.zeros(self.max_fields, dtype=np.int8)
        return observation, reward, done, False, {}

    def render(self) -> None:  # pragma: no cover
        print(f"Field status: {self.field_status}")

    def close(self) -> None:  # pragma: no cover
        pass


def _demo_run() -> None:
    """Run a demo episode using random actions."""
    forms = [
        FormDefinition(fields={"name": "Alice", "age": "30"}),
        FormDefinition(fields={"username": "user1", "password": "pass"}),
    ]
    env = FormNavigationEnv(forms, max_steps=5, random_seed=42)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    print("Demo episode total reward:", total_reward)


if __name__ == "__main__":  # pragma: no cover
    _demo_run()