from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Dataset/scripts"))

from prune_library_archives import VERSION_ROLES, package_roles  # noqa: E402


class LibraryVersionRoleTests(unittest.TestCase):
    def setUp(self) -> None:
        manifests = REPOSITORY / "Dataset/manifests"
        self.source_manifest = json.loads(
            (manifests / "source_manifest.json").read_text(encoding="utf-8")
        )
        self.role_manifest = json.loads(
            (manifests / "library_version_roles.json").read_text(encoding="utf-8")
        )
        self.matrix_path = manifests / "library_matrix.tsv"

    def test_every_source_version_has_one_explicit_semantic_role(self) -> None:
        packages = self.source_manifest["historical_packages"]
        resolved = package_roles(packages, self.role_manifest)

        self.assertEqual(len(resolved), len(packages))
        for package in packages:
            self.assertEqual(set(resolved[package["name"]]), set(VERSION_ROLES))

    def test_known_ordering_exceptions_are_classified_semantically(self) -> None:
        assignments = self.role_manifest["packages"]
        self.assertEqual(
            assignments["fribidi"]["minor-alternative"], "1.0.12"
        )
        self.assertEqual(
            assignments["fribidi"]["major-alternative"], "0.19.7-5"
        )
        self.assertEqual(assignments["gpm"]["minor-alternative"], "1.20.7-10")
        self.assertEqual(assignments["gpm"]["major-alternative"], "1.20.7-6")

    def test_generated_matrix_uses_only_explicit_roles(self) -> None:
        with self.matrix_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))

        self.assertEqual(len(rows), 169 * 32)
        self.assertEqual({row["role"] for row in rows}, set(VERSION_ROLES))
        self.assertNotIn("historical-1", {row["role"] for row in rows})
        self.assertNotIn("historical-2", {row["role"] for row in rows})


if __name__ == "__main__":
    unittest.main()
