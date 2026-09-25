"""Deterministic plan generation for training, serialization, and the pilot.

Stdlib-only. **This module never executes anything.** It imports no
subprocess, os.system, shutil, or network facility, and it writes no files --
that absence is a structural guarantee, not a convention, and
tests/test_orchestration.py asserts it by inspecting this module's own
imports. A plan is data describing what *would* run; turning a plan into
execution is deliberately somebody else's job, on a host that is not this one.

Authorization: build_pilot_plan() requires a PreflightReport whose every
precondition is READY, and raises NotAuthorized otherwise. To inspect what a
plan *would* look like while still blocked, call describe_blocked_plan(),
which returns the same steps annotated with the blockers and an explicit
executable=False marker. The two entry points exist so that "show me the
plan" can never be mistaken for "the plan is cleared to run".
"""

from __future__ import annotations

from dataclasses import dataclass

from medium_pc_audit.experiment import ArmSpec, ExperimentSpec
from medium_pc_audit.identity import compute_identity_sha256
from medium_pc_audit.preflight import PreflightReport, require_authorized

ACTION_TRAIN = "train"
ACTION_SERIALIZE_PT = "serialize_pt"
ACTION_SERIALIZE_NNUE = "serialize_nnue"
ACTION_ENGINE_LOAD_CHECK = "engine_load_check"
ACTION_MATCH = "match"
ACTION_PACKAGE = "package"

ACTIONS = frozenset(
    {
        ACTION_TRAIN,
        ACTION_SERIALIZE_PT,
        ACTION_SERIALIZE_NNUE,
        ACTION_ENGINE_LOAD_CHECK,
        ACTION_MATCH,
        ACTION_PACKAGE,
    }
)


class OrchestrationError(ValueError):
    """Raised when a plan cannot be constructed."""


@dataclass(frozen=True)
class PlannedStep:
    """One step of a plan. `command` is descriptive, never executed."""

    step_id: str
    action: str
    description: str
    command: tuple
    produces: tuple = ()

    def __post_init__(self):
        if not isinstance(self.step_id, str) or not self.step_id:
            raise OrchestrationError("step_id must be a non-empty string")

        if self.action not in ACTIONS:
            raise OrchestrationError(
                f"action must be one of {sorted(ACTIONS)}, got {self.action!r}"
            )

        object.__setattr__(self, "command", tuple(str(c) for c in self.command))
        object.__setattr__(self, "produces", tuple(str(p) for p in self.produces))

    def as_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "description": self.description,
            "command": list(self.command),
            "produces": list(self.produces),
        }


@dataclass(frozen=True)
class Plan:
    plan_id: str
    steps: tuple
    executable: bool
    blockers: tuple = ()

    def as_dict(self) -> dict:
        payload = {
            "plan_id": self.plan_id,
            "executable": self.executable,
            "steps": [s.as_dict() for s in self.steps],
            "blockers": [dict(b) for b in self.blockers],
        }
        payload["plan_identity_sha256"] = compute_identity_sha256(
            {
                "plan_id": self.plan_id,
                "steps": payload["steps"],
            }
        )
        return payload

    @property
    def step_count(self) -> int:
        return len(self.steps)


def build_training_plan(arm, *, workspace: str = "WORKSPACE") -> tuple:
    """Describe the train -> serialize -> engine-load-check steps for one arm.

    Every numeric hyperparameter is taken from the arm's shared TrainingBudget,
    so two arms of one experiment necessarily produce identical training
    commands apart from their data source.
    """

    if not isinstance(arm, ArmSpec):
        raise OrchestrationError(f"arm must be an ArmSpec, got {type(arm).__name__}")

    if not isinstance(workspace, str) or not workspace:
        raise OrchestrationError("workspace must be a non-empty string")

    budget = arm.budget
    base = f"{workspace}/{arm.arm_id}"

    steps = []

    for block in range(1, budget.blocks + 1):
        parent = (
            f"{arm.seed.artifact_id}.pt"
            if block == 1
            else f"{base}/block{block - 1:03d}/challenger.pt"
        )

        block_dir = f"{base}/block{block:03d}"

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-block{block:03d}-train",
                action=ACTION_TRAIN,
                description=(
                    f"arm {arm.arm_id!r} block {block}: fine-tune from {parent} "
                    f"on corpus {arm.data_source.source_id!r}"
                ),
                command=(
                    "python",
                    "train.py",
                    "--resume-from-model",
                    parent,
                    "--lr",
                    repr(budget.lr),
                    "--epoch-size",
                    str(budget.epoch_size),
                    "--batch-size",
                    str(budget.batch_size),
                    "--max-epochs",
                    "1",
                    "--optimizer-name",
                    budget.optimizer_name,
                    "--lambda",
                    repr(float(budget.lambda_)),
                    # Training datasets are POSITIONAL in nnue-pytorch's
                    # train.py ("[[STR [STR ...]]] Training datasets
                    # (.binpack)"), verified against the pinned trainer's own
                    # --help. There is no --training-dataset flag.
                    f"{arm.data_source.source_id}.binpack",
                ),
                produces=(f"{block_dir}/last.ckpt",),
            )
        )

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-block{block:03d}-serialize-pt",
                action=ACTION_SERIALIZE_PT,
                description=f"arm {arm.arm_id!r} block {block}: checkpoint -> .pt",
                command=(
                    "python",
                    "serialize.py",
                    f"{block_dir}/last.ckpt",
                    f"{block_dir}/challenger.pt",
                ),
                produces=(f"{block_dir}/challenger.pt",),
            )
        )

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-block{block:03d}-serialize-nnue",
                action=ACTION_SERIALIZE_NNUE,
                description=f"arm {arm.arm_id!r} block {block}: checkpoint -> .nnue",
                command=(
                    "python",
                    "serialize.py",
                    f"{block_dir}/last.ckpt",
                    f"{block_dir}/challenger.nnue",
                ),
                produces=(f"{block_dir}/challenger.nnue",),
            )
        )

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-block{block:03d}-engine-load",
                action=ACTION_ENGINE_LOAD_CHECK,
                description=(
                    f"arm {arm.arm_id!r} block {block}: confirm the engine loads "
                    "the produced network and completes a UCI handshake"
                ),
                command=(
                    "stockfish",
                    "setoption",
                    "name",
                    "EvalFile",
                    "value",
                    f"{block_dir}/challenger.nnue",
                ),
                produces=(f"{block_dir}/engine_load_check.txt",),
            )
        )

    return tuple(steps)


def build_match_plan(
    experiment,
    *,
    pairs: int,
    opening_suite_id: str,
    workspace: str = "WORKSPACE",
) -> tuple:
    """Describe the paired, color-swapped match and packaging steps per arm."""

    if not isinstance(experiment, ExperimentSpec):
        raise OrchestrationError("experiment must be an ExperimentSpec")

    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise OrchestrationError(f"pairs must be an int >= 1, got {pairs!r}")

    if not isinstance(opening_suite_id, str) or not opening_suite_id:
        raise OrchestrationError("opening_suite_id must be a non-empty string")

    steps = []

    for arm in sorted(experiment.arms, key=lambda a: a.arm_id):
        final_block = f"{workspace}/{arm.arm_id}/block{arm.budget.blocks:03d}"

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-match",
                action=ACTION_MATCH,
                description=(
                    f"arm {arm.arm_id!r}: {pairs} paired color-swapped games "
                    f"({pairs * 2} total) from suite {opening_suite_id!r}"
                ),
                command=(
                    "c-chess-cli",
                    "-engine",
                    f"name={arm.arm_id}",
                    f"option.EvalFile={final_block}/challenger.nnue",
                    "-engine",
                    "name=ANCHOR",
                    "-games",
                    str(pairs * 2),
                    "-repeat",
                    "-openings",
                    f"file={opening_suite_id}.epd",
                    "order=sequential",
                    "-pgn",
                    f"{workspace}/{arm.arm_id}/games.pgn",
                    "2",
                ),
                produces=(
                    f"{workspace}/{arm.arm_id}/games.pgn",
                    f"{workspace}/{arm.arm_id}/match.log",
                ),
            )
        )

        steps.append(
            PlannedStep(
                step_id=f"{arm.arm_id}-package",
                action=ACTION_PACKAGE,
                description=(
                    f"arm {arm.arm_id!r}: emit execution_manifest.json, "
                    "raw_result.json, artifact_binding.json and SHA256SUMS.txt "
                    "recording OBSERVED values on the execution host "
                    "for later evidence verification"
                ),
                command=(
                    "<execution-host packager>",
                    f"--run-dir={workspace}/{arm.arm_id}",
                ),
                produces=(
                    f"{workspace}/{arm.arm_id}/execution_manifest.json",
                    f"{workspace}/{arm.arm_id}/raw_result.json",
                    f"{workspace}/{arm.arm_id}/artifact_binding.json",
                    f"{workspace}/{arm.arm_id}/SHA256SUMS.txt",
                ),
            )
        )

    return tuple(steps)


def _assemble_steps(experiment, *, pairs, opening_suite_id, workspace) -> tuple:
    steps = []

    for arm in sorted(experiment.arms, key=lambda a: a.arm_id):
        steps.extend(build_training_plan(arm, workspace=workspace))

    steps.extend(
        build_match_plan(
            experiment,
            pairs=pairs,
            opening_suite_id=opening_suite_id,
            workspace=workspace,
        )
    )

    return tuple(steps)


def build_pilot_plan(
    experiment,
    report,
    *,
    pairs: int,
    opening_suite_id: str,
    workspace: str = "WORKSPACE",
) -> Plan:
    """Build an executable pilot plan, or refuse.

    Raises NotAuthorized unless every preflight precondition is READY. There
    is deliberately no override parameter.
    """

    if not isinstance(report, PreflightReport):
        raise OrchestrationError("report must be a PreflightReport")

    if not isinstance(experiment, ExperimentSpec):
        raise OrchestrationError("experiment must be an ExperimentSpec")

    if report.experiment_identity_sha256 != experiment.identity_sha256:
        raise OrchestrationError(
            "preflight report does not correspond to this experiment "
            f"({report.experiment_identity_sha256[:12]}... vs "
            f"{experiment.identity_sha256[:12]}...); a report from a different "
            "specification cannot authorize this one"
        )

    require_authorized(report)

    return Plan(
        plan_id=f"{experiment.experiment_id}-pilot",
        steps=_assemble_steps(
            experiment,
            pairs=pairs,
            opening_suite_id=opening_suite_id,
            workspace=workspace,
        ),
        executable=True,
        blockers=(),
    )


def describe_blocked_plan(
    experiment,
    report,
    *,
    pairs: int,
    opening_suite_id: str,
    workspace: str = "WORKSPACE",
) -> Plan:
    """Return the same steps for inspection, explicitly marked not executable.

    Always returns executable=False regardless of the report's contents, so
    this function can never be mistaken for an authorization path.
    """

    if not isinstance(report, PreflightReport):
        raise OrchestrationError("report must be a PreflightReport")

    return Plan(
        plan_id=f"{experiment.experiment_id}-pilot-DRAFT",
        steps=_assemble_steps(
            experiment,
            pairs=pairs,
            opening_suite_id=opening_suite_id,
            workspace=workspace,
        ),
        executable=False,
        blockers=tuple(p.as_dict() for p in report.blockers),
    )
