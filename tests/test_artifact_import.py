"""Tests for medium_pc_audit.artifact_import -- Phase 1.

All artifacts used here are synthetic and temporary. No real BigPC data,
no network access, no external dependency.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medium_pc_audit import artifact_import
from medium_pc_audit.schemas import registry
from tests.fixtures.synthetic_source_dirs import make_source_dir

FIXED_NOW = "2026-09-22T12:00:00Z"


class ArtifactImportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.dest_root = self.base / "dest"
        self.dest_root.mkdir()

    def seal_default(self, artifact_id="FIXTURE_ART", source_dir=None):
        source_dir = source_dir or make_source_dir(self.base)
        return artifact_import.seal(
            artifact_type="opening_suite",
            artifact_id=artifact_id,
            source_dir=source_dir,
            dest_root=self.dest_root,
            source_description="synthetic fixture",
            imported_by="tester",
            now_utc=FIXED_NOW,
        )

    def default_final_dir(self):
        return self.dest_root / "opening_suite" / "FIXTURE_ART"


class SealHappyPathTests(ArtifactImportTestCase):
    def test_seal_creates_manifest_files_and_sums(self):
        source_dir = make_source_dir(self.base)
        manifest = self.seal_default(source_dir=source_dir)

        final_dir = self.default_final_dir()
        self.assertTrue((final_dir / "manifest.json").is_file())
        self.assertTrue((final_dir / "SHA256SUMS.txt").is_file())
        self.assertTrue((final_dir / "files" / "readme.txt").is_file())
        self.assertTrue((final_dir / "files" / "sub" / "data.bin").is_file())

        registry.validate("artifact_manifest", "v1", manifest)
        self.assertTrue(manifest["sealed"])
        self.assertEqual(manifest["artifact_id"], "FIXTURE_ART")

    def test_verify_seal_passes_immediately_after_seal(self):
        self.seal_default()
        artifact_import.verify_seal(self.default_final_dir())

    def test_sealed_files_are_not_marked_read_only(self):
        self.seal_default()
        readme = self.default_final_dir() / "files" / "readme.txt"
        self.assertTrue(
            os.access(readme, os.W_OK),
            "Phase 1 must not use read-only permissions as an integrity mechanism",
        )


class SealRejectionTests(ArtifactImportTestCase):
    def test_seal_rejects_existing_final_directory(self):
        self.seal_default()
        with self.assertRaises(artifact_import.ArtifactAlreadySealed):
            self.seal_default()

        artifact_import.verify_seal(self.default_final_dir())

    def test_seal_rejects_empty_source_dir(self):
        empty = self.base / "empty_source"
        empty.mkdir()
        with self.assertRaises(artifact_import.EmptyArtifactSource):
            self.seal_default(source_dir=empty)

    def test_seal_rejects_nonexistent_source_dir(self):
        with self.assertRaises(artifact_import.EmptyArtifactSource):
            self.seal_default(source_dir=self.base / "does_not_exist")

    def test_seal_rejects_invalid_artifact_type(self):
        source_dir = make_source_dir(self.base)
        with self.assertRaises(artifact_import.InvalidArtifactType):
            artifact_import.seal(
                artifact_type="not_a_real_type",
                artifact_id="X",
                source_dir=source_dir,
                dest_root=self.dest_root,
                source_description="x",
                imported_by="tester",
                now_utc=FIXED_NOW,
            )

    def test_seal_rejects_unsafe_artifact_id(self):
        source_dir = make_source_dir(self.base)
        bad_ids = ["", ".", "..", "a/b", "a\\b", "C:", "a:b", "a b", "../escape"]
        for bad_id in bad_ids:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(artifact_import.UnsafeArtifactId):
                    artifact_import.seal(
                        artifact_type="opening_suite",
                        artifact_id=bad_id,
                        source_dir=source_dir,
                        dest_root=self.dest_root,
                        source_description="x",
                        imported_by="tester",
                        now_utc=FIXED_NOW,
                    )

    def test_seal_rejects_source_destination_overlap_source_inside_dest(self):
        overlapping_source = self.dest_root / "inside"
        overlapping_source.mkdir()
        (overlapping_source / "f.txt").write_bytes(b"x")
        with self.assertRaises(artifact_import.OverlappingArtifactPaths):
            self.seal_default(source_dir=overlapping_source)

    def test_seal_rejects_source_destination_overlap_dest_inside_source(self):
        outer_source = self.base / "outer_source"
        outer_source.mkdir()
        (outer_source / "f.txt").write_bytes(b"x")
        nested_dest = outer_source / "nested_dest"
        nested_dest.mkdir()
        with self.assertRaises(artifact_import.OverlappingArtifactPaths):
            artifact_import.seal(
                artifact_type="opening_suite",
                artifact_id="X",
                source_dir=outer_source,
                dest_root=nested_dest,
                source_description="x",
                imported_by="tester",
                now_utc=FIXED_NOW,
            )


@unittest.skipUnless(hasattr(os, "symlink"), "platform has no os.symlink")
class SymlinkRejectionTests(ArtifactImportTestCase):
    def _try_make_symlink(self, link_path, target_path, target_is_directory=False):
        try:
            os.symlink(target_path, link_path, target_is_directory=target_is_directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symlinks in this environment: {exc}")

    def test_seal_rejects_symlinked_file(self):
        source_dir = make_source_dir(self.base)
        real_target = self.base / "outside_target.bin"
        real_target.write_bytes(b"target content")
        link_path = source_dir / "linked_file.bin"
        self._try_make_symlink(link_path, real_target)

        with self.assertRaises(artifact_import.UnsafeSourceEntry):
            self.seal_default(source_dir=source_dir)

    def test_seal_rejects_symlinked_directory(self):
        source_dir = make_source_dir(self.base)
        real_target_dir = self.base / "outside_target_dir"
        real_target_dir.mkdir()
        (real_target_dir / "inner.txt").write_bytes(b"inner")
        link_path = source_dir / "linked_dir"
        self._try_make_symlink(link_path, real_target_dir, target_is_directory=True)

        with self.assertRaises(artifact_import.UnsafeSourceEntry):
            self.seal_default(source_dir=source_dir)


class VerifySealDetectionTests(ArtifactImportTestCase):
    def test_detects_tampered_file(self):
        self.seal_default()
        target = self.default_final_dir() / "files" / "readme.txt"
        target.write_bytes(target.read_bytes() + b"TAMPERED")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("readme.txt" in e for e in ctx.exception.errors))

    def test_detects_missing_file(self):
        self.seal_default()
        (self.default_final_dir() / "files" / "readme.txt").unlink()

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("readme.txt" in e and "missing" in e for e in ctx.exception.errors))

    def test_detects_unexpected_extra_file(self):
        self.seal_default()
        (self.default_final_dir() / "files" / "unexpected.txt").write_bytes(b"not in manifest")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("unexpected.txt" in e and "extra" in e for e in ctx.exception.errors))

    def test_detects_manifest_json_corruption(self):
        self.seal_default()
        (self.default_final_dir() / "manifest.json").write_text("{not valid json", encoding="utf-8")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("not valid JSON" in e for e in ctx.exception.errors))

    def test_detects_manifest_schema_violation(self):
        self.seal_default()
        manifest_path = self.default_final_dir() / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["sealed"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("schema validation" in e for e in ctx.exception.errors))

    def test_detects_sha256sums_mismatch(self):
        self.seal_default()
        sums_path = self.default_final_dir() / "SHA256SUMS.txt"
        sums_path.write_text(sums_path.read_text(encoding="utf-8") + "extra-line\n", encoding="utf-8")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("SHA256SUMS.txt" in e for e in ctx.exception.errors))


class IdentityBindingTests(ArtifactImportTestCase):
    """Hardening: verify_seal() must bind manifest identity to the
    filesystem location it was found at, not just trust the manifest.
    """

    def _mutate_manifest(self, mutator):
        manifest_path = self.default_final_dir() / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutator(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_detects_artifact_id_mismatch(self):
        self.seal_default()
        self._mutate_manifest(lambda m: m.__setitem__("artifact_id", "SOMETHING_ELSE"))

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("artifact_id" in e for e in ctx.exception.errors))

    def test_detects_artifact_type_mismatch(self):
        self.seal_default()
        # "champion_v1_anchor" is a valid enum member -- just the wrong one
        # for a directory actually sitting under opening_suite/.
        self._mutate_manifest(lambda m: m.__setitem__("artifact_type", "champion_v1_anchor"))

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("artifact_type" in e for e in ctx.exception.errors))

    def test_detects_sealed_not_true(self):
        self.seal_default()
        self._mutate_manifest(lambda m: m.__setitem__("sealed", False))

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("sealed" in e for e in ctx.exception.errors))


class DuplicatePathTests(ArtifactImportTestCase):
    def test_detects_duplicate_paths_even_with_identical_hash_and_size(self):
        self.seal_default()
        manifest_path = self.default_final_dir() / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Duplicate one existing entry verbatim -- identical hash/size is
        # exactly the subtle case that must still be rejected.
        manifest["files"].append(dict(manifest["files"][0]))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("duplicate" in e.lower() for e in ctx.exception.errors))


class TopLevelLayoutTests(ArtifactImportTestCase):
    def test_detects_unexpected_top_level_file(self):
        self.seal_default()
        (self.default_final_dir() / "EXTRA.txt").write_bytes(b"should not be here")

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("EXTRA.txt" in e for e in ctx.exception.errors))

    def test_detects_unexpected_top_level_directory(self):
        self.seal_default()
        (self.default_final_dir() / "extra_dir").mkdir()

        with self.assertRaises(artifact_import.ArtifactIntegrityError) as ctx:
            artifact_import.verify_seal(self.default_final_dir())
        self.assertTrue(any("extra_dir" in e for e in ctx.exception.errors))


class DeterministicOrderingTests(ArtifactImportTestCase):
    def test_manifest_and_sums_sorted_regardless_of_creation_order(self):
        source_dir = self.base / "source"
        source_dir.mkdir()
        (source_dir / "z_first_created.txt").write_bytes(b"z")
        (source_dir / "a_second_created.txt").write_bytes(b"a")
        (source_dir / "m_third_created.txt").write_bytes(b"m")

        manifest = self.seal_default(source_dir=source_dir)
        paths_in_manifest = [entry["path"] for entry in manifest["files"]]
        self.assertEqual(paths_in_manifest, sorted(paths_in_manifest))

        sums_lines = (self.default_final_dir() / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        sums_paths = [line.split("  ", 1)[1] for line in sums_lines]
        self.assertEqual(sums_paths, sorted(sums_paths))
        self.assertEqual(sums_paths, paths_in_manifest)


class ImportArtifactTests(ArtifactImportTestCase):
    def test_import_seals_when_new(self):
        source_dir = make_source_dir(self.base)
        manifest = artifact_import.import_artifact(
            artifact_type="opening_suite",
            artifact_id="NEW_ART",
            source_dir=source_dir,
            dest_root=self.dest_root,
            source_description="synthetic",
            imported_by="tester",
            now_utc=FIXED_NOW,
        )
        self.assertEqual(manifest["artifact_id"], "NEW_ART")

    def test_import_identical_content_is_idempotent_noop(self):
        source_dir = make_source_dir(self.base)
        first = artifact_import.import_artifact(
            artifact_type="opening_suite",
            artifact_id="ART",
            source_dir=source_dir,
            dest_root=self.dest_root,
            source_description="synthetic",
            imported_by="tester",
            now_utc=FIXED_NOW,
        )
        second = artifact_import.import_artifact(
            artifact_type="opening_suite",
            artifact_id="ART",
            source_dir=source_dir,
            dest_root=self.dest_root,
            source_description="synthetic (re-import)",
            imported_by="tester2",
            now_utc="2099-01-01T00:00:00Z",
        )
        self.assertEqual(first, second)

    def test_import_differing_content_raises_duplicate_conflict(self):
        source_dir = make_source_dir(self.base)
        artifact_import.import_artifact(
            artifact_type="opening_suite",
            artifact_id="ART",
            source_dir=source_dir,
            dest_root=self.dest_root,
            source_description="synthetic",
            imported_by="tester",
            now_utc=FIXED_NOW,
        )

        changed_source = make_source_dir(self.base / "changed", files={"readme.txt": b"different content entirely"})
        with self.assertRaises(artifact_import.DuplicateArtifactConflict) as ctx:
            artifact_import.import_artifact(
                artifact_type="opening_suite",
                artifact_id="ART",
                source_dir=changed_source,
                dest_root=self.dest_root,
                source_description="synthetic",
                imported_by="tester",
                now_utc=FIXED_NOW,
            )
        self.assertTrue(any("readme.txt" in e for e in ctx.exception.errors))


class StagedCopyMismatchTests(ArtifactImportTestCase):
    def test_staged_copy_mismatch_detected_and_leaves_no_final_directory(self):
        source_dir = make_source_dir(self.base)
        real_copy = artifact_import._atomic_copy
        state = {"calls": 0}

        def corrupting_copy(src, dst):
            real_copy(src, dst)
            state["calls"] += 1
            if state["calls"] == 1:
                with open(dst, "ab") as f:
                    f.write(b"CORRUPTION")

        with mock.patch.object(artifact_import, "_atomic_copy", side_effect=corrupting_copy):
            with self.assertRaises(artifact_import.StagedCopyMismatch):
                self.seal_default(source_dir=source_dir)

        self.assertFalse(self.default_final_dir().exists())

        type_root = self.dest_root / "opening_suite"
        leftover_staging = [p for p in type_root.iterdir() if p.name.startswith(".staging-")]
        self.assertEqual(leftover_staging, [], "failed seal must not leave a staging directory behind")


class PublicationRaceTests(ArtifactImportTestCase):
    def test_final_directory_appearing_during_staging_is_never_overwritten(self):
        source_dir = make_source_dir(self.base)
        final_dir = self.default_final_dir()

        real_write = artifact_import._atomic_write_text
        state = {"calls": 0}

        def racing_write(path, text):
            state["calls"] += 1
            # Simulate a rival process publishing this exact artifact
            # between our own staging work and our own publish step
            # (right after we finish writing manifest.json, before we
            # write SHA256SUMS.txt and reach the pre-publish existence
            # check).
            if state["calls"] == 2 and not final_dir.exists():
                final_dir.mkdir(parents=True)
                (final_dir / "manifest.json").write_text('{"rival": true}', encoding="utf-8")
                (final_dir / "RIVAL_MARKER.txt").write_text("rival content", encoding="utf-8")
            real_write(path, text)

        with mock.patch.object(artifact_import, "_atomic_write_text", side_effect=racing_write):
            with self.assertRaises(artifact_import.ArtifactAlreadySealed):
                self.seal_default(source_dir=source_dir)

        # The rival's content must be completely untouched by our failed attempt.
        self.assertEqual(
            (final_dir / "manifest.json").read_text(encoding="utf-8"),
            '{"rival": true}',
        )
        self.assertTrue((final_dir / "RIVAL_MARKER.txt").is_file())

        # Our own failed attempt must not leave any stray staging directory behind.
        type_root = self.dest_root / "opening_suite"
        leftover = [p for p in type_root.iterdir() if p.name.startswith(".staging-")]
        self.assertEqual(leftover, [], "failed seal must not leave a staging directory behind")


class NoMutatingApiTests(unittest.TestCase):
    def test_no_update_or_delete_or_force_overwrite_api(self):
        for forbidden_name in ("update_artifact", "delete_artifact", "force_reseal", "overwrite_artifact"):
            self.assertFalse(
                hasattr(artifact_import, forbidden_name),
                f"artifact_import must not expose {forbidden_name}",
            )


if __name__ == "__main__":
    unittest.main()
