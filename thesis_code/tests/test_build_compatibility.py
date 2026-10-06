from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Dataset/scripts"))

from build_randomized_elf_matrix import configure_cache_overrides  # noqa: E402


class BuildCompatibilityTests(unittest.TestCase):
    def test_bash_uses_fallback_with_glibc_without_sys_random_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            self.assertEqual(
                configure_cache_overrides("bash", {"glibc": prefix}),
                {
                    "ac_cv_header_sys_random_h": "no",
                    "ac_cv_func_getrandom": "no",
                },
            )

    def test_bash_keeps_normal_probes_when_selected_glibc_has_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            header = prefix / "include/sys/random.h"
            header.parent.mkdir(parents=True)
            header.touch()
            self.assertEqual(
                configure_cache_overrides("bash", {"glibc": prefix}),
                {},
            )

    def test_acl_projects_ignore_build_filesystem_runtime_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            for program in ("coreutils", "sed", "tar"):
                with self.subTest(program=program):
                    self.assertEqual(
                        configure_cache_overrides(
                            program,
                            {"glibc": prefix, "acl": prefix},
                        ),
                        {"gl_cv_func_working_acl_get_file": "yes"},
                    )

    def test_unrelated_programs_are_not_affected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            self.assertEqual(
                configure_cache_overrides("less", {"glibc": prefix}),
                {},
            )


if __name__ == "__main__":
    unittest.main()
