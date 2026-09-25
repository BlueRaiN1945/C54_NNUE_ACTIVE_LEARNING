"""Tests for plan generation, including its structural inability to execute."""

import ast
import inspect
import unittest

from medium_pc_audit import orchestration
from medium_pc_audit.experiment import (
    EVIDENCE_IMPORT_RECORD,
    EvidenceRef,
    build_experiment,
)
from medium_pc_audit.orchestration import (
    ACTION_MATCH,
    ACTION_PACKAGE,
    ACTION_TRAIN,
    OrchestrationError,
    build_match_plan,
    build_pilot_plan,
    build_training_plan,
    describe_blocked_plan,
)
from medium_pc_audit.preflight import NotAuthorized, evaluate_preflight
from tests.test_experiment import (
    SEED_A,
    budget,
    control_arm,
    creation_evidence,
    roundtrip_evidence,
    seed,
    training_evidence,
    treatment_arm,
)
from tests.test_preflight import (
    evidenced_free_source,
    experiment,
    good_suite,
    isolated_split,
    proven_seed,
)


def authorized_experiment(seed_spec=None):
    s = seed_spec or proven_seed()

    return build_experiment(
        experiment_id="EXP1",
        arms=[
            control_arm(
                seed=s,
                data_source=evidenced_free_source(),
            ),
            treatment_arm(seed=s),
        ],
    )

def authorized_report(exp):
    return evaluate_preflight(
        exp,
        opening_suite=good_suite(50),
        split_result=isolated_split(),
        pairs=20,
    )


class NoExecutionCapabilityTests(unittest.TestCase):
    """The strongest guarantee here is structural, so it is asserted directly.

    These assertions parse the module's AST rather than scanning its text: a
    docstring that *names* subprocess in order to disclaim it must not trip
    the guard, while an actual import must. Checking what the module does,
    not what it says, is the whole point.
    """

    FORBIDDEN_MODULES = frozenset(
        {
            "subprocess",
            "os",
            "shutil",
            "socket",
            "urllib",
            "requests",
            "http",
            "asyncio",
            "multiprocessing",
            "pty",
            "popen2",
            "commands",
        }
    )

    FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "open", "__import__"})

    def module_tree(self):
        return ast.parse(inspect.getsource(orchestration))

    def test_imports_no_execution_or_network_module(self):
        imported = set()

        for node in ast.walk(self.module_tree()):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        offending = imported & self.FORBIDDEN_MODULES

        self.assertEqual(
            offending,
            set(),
            f"orchestration.py must not import {sorted(offending)}",
        )

    def test_calls_no_execution_builtin(self):
        called = set()

        for node in ast.walk(self.module_tree()):
            if isinstance(node, ast.Call):
                func = node.func

                if isinstance(func, ast.Name):
                    called.add(func.id)
                elif isinstance(func, ast.Attribute):
                    called.add(func.attr)

        offending = called & self.FORBIDDEN_CALLS

        self.assertEqual(
            offending,
            set(),
            f"orchestration.py must not call {sorted(offending)}",
        )

    def test_docstring_disclaimer_does_not_defeat_the_guard(self):
        """Guard against the guard: the disclaimer text is present on purpose."""

        self.assertIn("subprocess", inspect.getsource(orchestration))


class TrainingPlanTests(unittest.TestCase):
    def test_one_block_produces_four_steps(self):
        steps = build_training_plan(control_arm())

        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0].action, ACTION_TRAIN)

    def test_block_count_scales_steps(self):
        arm = control_arm(budget=budget(blocks=3))

        self.assertEqual(len(build_training_plan(arm)), 12)

    def test_first_block_resumes_from_the_seed(self):
        steps = build_training_plan(control_arm())

        self.assertIn("--resume-from-model", steps[0].command)
        idx = steps[0].command.index("--resume-from-model")
        self.assertIn("OFFICIAL_BASE", steps[0].command[idx + 1])

    def test_later_block_chains_from_previous_challenger(self):
        steps = build_training_plan(control_arm(budget=budget(blocks=2)))
        second_train = steps[4]

        idx = second_train.command.index("--resume-from-model")
        self.assertIn("block001/challenger.pt", second_train.command[idx + 1])

    def test_budget_values_are_carried_into_the_command(self):
        steps = build_training_plan(control_arm())
        command = steps[0].command

        self.assertIn("65536", command)
        self.assertIn("4096", command)
        self.assertIn("rangerlite", command)

    def test_arms_sharing_a_budget_differ_only_by_data_and_name(self):
        s = seed()
        control = build_training_plan(control_arm(seed=s))[0].command
        treatment = build_training_plan(treatment_arm(seed=s))[0].command

        differing = [(a, b) for a, b in zip(control, treatment) if a != b]

        # Only the corpus differs; every hyperparameter token matches.
        self.assertEqual(len(differing), 1)
        self.assertIn("binpack", differing[0][0])

    def test_non_arm_rejected(self):
        with self.assertRaises(OrchestrationError):
            build_training_plan("not-an-arm")


class MatchPlanTests(unittest.TestCase):
    def test_two_steps_per_arm(self):
        steps = build_match_plan(experiment(), pairs=20, opening_suite_id="UHO")

        self.assertEqual(len(steps), 4)
        self.assertEqual({s.action for s in steps}, {ACTION_MATCH, ACTION_PACKAGE})

    def test_game_count_is_twice_the_pairs(self):
        steps = build_match_plan(experiment(), pairs=20, opening_suite_id="UHO")
        match_step = next(s for s in steps if s.action == ACTION_MATCH)

        self.assertIn("40", match_step.command)

    def test_match_is_paired_and_sequential(self):
        steps = build_match_plan(experiment(), pairs=5, opening_suite_id="UHO")
        match_step = next(s for s in steps if s.action == ACTION_MATCH)

        self.assertIn("-repeat", match_step.command)
        self.assertIn("order=sequential", match_step.command)
        self.assertNotIn("order=random", match_step.command)

    def test_package_step_produces_contract_files(self):
        steps = build_match_plan(experiment(), pairs=5, opening_suite_id="UHO")
        package = next(s for s in steps if s.action == ACTION_PACKAGE)

        for required in (
            "execution_manifest.json",
            "raw_result.json",
            "artifact_binding.json",
            "SHA256SUMS.txt",
        ):
            self.assertTrue(
                any(required in p for p in package.produces),
                f"package step must produce {required}",
            )

    def test_invalid_pairs_rejected(self):
        with self.assertRaises(OrchestrationError):
            build_match_plan(experiment(), pairs=0, opening_suite_id="UHO")


class AuthorizationTests(unittest.TestCase):
    def test_blocked_preflight_refuses_to_build_a_pilot(self):
        exp = experiment()

        with self.assertRaises(NotAuthorized):
            build_pilot_plan(
                exp,
                evaluate_preflight(exp),
                pairs=20,
                opening_suite_id="UHO",
            )

    def test_authorized_preflight_builds_an_executable_plan(self):
        exp = authorized_experiment()
        plan = build_pilot_plan(
            exp,
            authorized_report(exp),
            pairs=20,
            opening_suite_id="UHO",
        )

        self.assertTrue(plan.executable)
        self.assertEqual(plan.blockers, ())
        self.assertEqual(plan.step_count, 12)

    def test_report_from_a_different_experiment_is_rejected(self):
        """A passing report cannot be transplanted onto another spec."""

        other = authorized_experiment()
        report = authorized_report(other)

        # Same claims, different evidence record -> different identity.
        mismatched = authorized_experiment(
            seed(
                evidence=(
                    roundtrip_evidence(),
                    creation_evidence(),
                    training_evidence(),
                    EvidenceRef(
                        kind=EVIDENCE_IMPORT_RECORD,
                        record_id="A_SECOND_IMPORT_RECORD",
                        subject_sha256=SEED_A,
                        verified_at="2026-09-24T09:00:00Z",
                    ),
                )
            )
        )

        with self.assertRaises(OrchestrationError) as ctx:
            build_pilot_plan(
                mismatched,
                report,
                pairs=20,
                opening_suite_id="UHO",
            )

        self.assertIn("does not correspond", str(ctx.exception))

    def test_plan_is_deterministic(self):
        exp = authorized_experiment()

        a = build_pilot_plan(
            exp, authorized_report(exp), pairs=20, opening_suite_id="UHO"
        ).as_dict()
        b = build_pilot_plan(
            exp, authorized_report(exp), pairs=20, opening_suite_id="UHO"
        ).as_dict()

        self.assertEqual(a["plan_identity_sha256"], b["plan_identity_sha256"])

    def test_non_report_rejected(self):
        with self.assertRaises(OrchestrationError):
            build_pilot_plan(
                experiment(), "not-a-report", pairs=1, opening_suite_id="UHO"
            )


class BlockedDescriptionTests(unittest.TestCase):
    def test_draft_is_never_executable(self):
        exp = experiment()
        plan = describe_blocked_plan(
            exp,
            evaluate_preflight(exp),
            pairs=20,
            opening_suite_id="UHO",
        )

        self.assertFalse(plan.executable)
        self.assertTrue(plan.blockers)
        self.assertIn("DRAFT", plan.plan_id)

    def test_draft_stays_non_executable_even_when_authorized(self):
        """describe_blocked_plan can never become an authorization path."""

        exp = authorized_experiment()
        plan = describe_blocked_plan(
            exp,
            authorized_report(exp),
            pairs=20,
            opening_suite_id="UHO",
        )

        self.assertFalse(plan.executable)

    def test_draft_shows_the_same_steps(self):
        exp = authorized_experiment()
        report = authorized_report(exp)

        executable = build_pilot_plan(exp, report, pairs=20, opening_suite_id="UHO")
        draft = describe_blocked_plan(exp, report, pairs=20, opening_suite_id="UHO")

        self.assertEqual(
            [s.step_id for s in executable.steps],
            [s.step_id for s in draft.steps],
        )


if __name__ == "__main__":
    unittest.main()
