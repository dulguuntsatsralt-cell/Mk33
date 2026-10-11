"""Agent loop with Round Ledger Compaction (paper Section 3.2).

Used both by the teacher during data collection and by the trained model at
inference. Contexts come from ``ContextBuilder``, which the training replay also
uses, so training contexts match test-time contexts exactly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..envs.spec import EnvSpec
from ..llm import ChatClient
from ..trajectory import Trajectory, Turn
from .context import SYSTEM_PROMPT, ContextBuilder
from .rlc import RLCConfig
from .tools import ToolExecutor


@dataclass
class Budget:
    max_tool_calls: int = 400
    max_wall_s: float = 12 * 3600
    max_rounds: int | None = None


@dataclass
class AgentConfig:
    budget: Budget = field(default_factory=Budget)
    rlc: RLCConfig = field(default_factory=RLCConfig)
    system_prompt: str = SYSTEM_PROMPT


class Agent:
    def __init__(self, client: ChatClient, executor: ToolExecutor, env: EnvSpec,
                 cfg: AgentConfig | None = None, count_tokens=None):
        self.client = client
        self.ex = executor
        self.env = env
        self.cfg = cfg or AgentConfig()
        self.ctx = ContextBuilder(
            env.task_statement, "\n\n".join(env.skills), self.cfg.system_prompt,
            self.cfg.rlc, count_tokens,
        )

    def run(self) -> Trajectory:
        env, budget = self.env, self.cfg.budget
        traj = Trajectory(env_id=env.env_id, domain=env.domain, task_statement=env.task_statement,
                          pinned={"system": self.cfg.system_prompt, "skills": "\n\n".join(env.skills)})
        if env.baseline_score is not None and env.reference_scores and len(env.reference_scores) > 1:
            import statistics

            from ..envs.normalize import normalized_noise
            traj.metadata["eps"] = normalized_noise(
                statistics.stdev(env.reference_scores), env.baseline_score, env.reference_score)
        t0 = time.time()
        compactions: list[dict] = []
        traj.metadata["stop"] = "tool_budget"

        for i in range(budget.max_tool_calls):
            if time.time() - t0 > budget.max_wall_s:
                traj.metadata["stop"] = "wall_budget"
                break
            if budget.max_rounds and self.ex.round > budget.max_rounds:
                traj.metadata["stop"] = "round_budget"
                break
            msgs, event = self.ctx.next_context()
            if event is not None:
                compactions.append({"turn": i, **asdict(event)})
            resp = self.client.chat(msgs, tools=self.ex.schemas())
            calls = resp.get("tool_calls") or []
            ts = time.time()
            result = None
            if not calls:
                tool, args = "none", {}
                obs = "[no tool call. Continue by calling exactly one tool.]"
            else:
                tool, args = calls[0]["name"], calls[0].get("arguments") or {}
                result = self.ex.call(tool, args)
                obs = result.observation
                if len(calls) > 1:
                    obs += f"\n[note: only the first of {len(calls)} tool calls was executed]"

            sub = result.submission if result else None
            turn = Turn(
                index=i, round=sub.round if (sub and tool == "submit") else self.ex.round,
                reasoning=resp.get("content") or "", tool=tool, args=args, observation=obs,
                workspace_hash=self.ex.workspace_hash(),
                diff=result.diff if result else None, submission=sub,
                timed_out=bool(result and result.timed_out),
                wall_time_s=time.time() - ts, elapsed_s=time.time() - t0,
            )
            entry = self.ctx.make_entry(turn)
            if entry is not None:  # show the new ledger line right away
                turn.observation += f"\n[ledger] {entry.render()}"
            traj.turns.append(turn)
            self.ctx.record(turn)
            if result and result.done:
                traj.metadata["stop"] = "answer"
                break

        traj.metadata["compactions"] = compactions
        traj.metadata["rlc"] = asdict(self.cfg.rlc)
        traj.metadata["wall_s"] = time.time() - t0
        return traj
