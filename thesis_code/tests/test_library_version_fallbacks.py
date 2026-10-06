from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Dataset/scripts"))

import build_randomized_elf_matrix as matrix  # noqa: E402


class LibraryVersionFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.variants = dict(matrix.MATRIX_LIBRARY_VARIANTS)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        matrix.MATRIX_LIBRARY_VARIANTS.clear()
        matrix.MATRIX_LIBRARY_VARIANTS.update(self.variants)
        self.temporary_directory.cleanup()

    def variant(
        self,
        source: str,
        role: str,
        compiler: str,
        optimization: str,
        archive_name: str = "libc.a",
    ) -> matrix.LibraryVariant:
        archive = self.root / source / compiler / optimization / archive_name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.touch()
        return matrix.LibraryVariant(
            source=source,
            role=role,
            compiler=compiler,
            optimization=optimization,
            archives=(archive,),
        )

    def test_fallback_changes_source_and_preserves_variant_cell(self) -> None:
        initial = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        current_same_cell = self.variant(
            "glibc-2.41", "current", "gcc-13", "Os"
        )
        self.variant("glibc-2.41", "current", "gcc-11", "O2")
        minor = self.variant(
            "glibc-2.39", "minor-alternative", "gcc-11", "O2"
        )
        matrix.MATRIX_LIBRARY_VARIANTS.clear()
        matrix.MATRIX_LIBRARY_VARIANTS["glibc"] = [
            initial,
            current_same_cell,
            minor,
        ]

        plans = matrix.alternate_version_plans(
            seed=7,
            program="bash",
            compiler="gcc-11",
            elf_optimization="O0",
            libraries=("glibc",),
            selected={"glibc": "Os"},
            archive_overrides={("glibc", "libc.a"): initial.archives[0]},
            forced={},
        )

        self.assertEqual(len(plans), 3)
        self.assertIsNone(plans[0].changed_library)
        self.assertEqual(plans[1].variant, current_same_cell)
        self.assertEqual(plans[2].variant, minor)
        self.assertEqual(plans[1].selected["glibc"], "Os")

    def test_libc_abi_failure_prioritizes_glibc(self) -> None:
        glibc = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        ncurses_archive = self.root / "ncurses-5.9" / "libtinfow.a"
        ncurses_archive.parent.mkdir(parents=True)
        ncurses_archive.touch()
        log = (
            f"/usr/bin/ld: {ncurses_archive}(getenv_num.o): in function x:\n"
            "getenv_num.c:(.text+0x37): undefined reference to "
            "`__isoc23_strtol'\n"
        )

        implicated = matrix.version_failure_libraries(
            log,
            ("glibc", "ncurses"),
            {
                ("glibc", "libc.a"): glibc.archives[0],
                ("ncurses", "libncursesw.a"): ncurses_archive,
                ("ncurses", "libtinfow.a"): ncurses_archive,
            },
        )

        self.assertEqual(implicated, ["glibc", "ncurses"])

    def test_lld_followup_context_identifies_archive(self) -> None:
        openssl_archive = self.root / "openssl-1.1.1w" / "libssl.a"
        openssl_archive.parent.mkdir(parents=True)
        openssl_archive.touch()
        crypto_archive = openssl_archive.with_name("libcrypto.a")
        crypto_archive.touch()
        log = (
            "ld.lld: error: undefined symbol: legacy_ssl_entrypoint\n"
            ">>> referenced by client.c:42\n"
            f">>>               {openssl_archive}(ssl_lib.o):(connect_ssl)\n"
            "clang: error: linker command failed with exit code 1\n"
        )

        implicated = matrix.version_failure_libraries(
            log,
            ("openssl",),
            {
                ("openssl", "libssl.a"): openssl_archive,
                ("openssl", "libcrypto.a"): crypto_archive,
            },
        )

        self.assertEqual(implicated, ["openssl"])

    def test_non_link_failure_does_not_trigger_version_retry(self) -> None:
        glibc = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        self.assertEqual(
            matrix.version_failure_libraries(
                "Failed to find cmarkgfm or commonmark for python3.\n",
                ("glibc",),
                {("glibc", "libc.a"): glibc.archives[0]},
            ),
            [],
        )

    def test_configure_version_error_identifies_named_library(self) -> None:
        glibc = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        openssl_archive = self.root / "openssl-1.1.1w" / "libssl.a"
        openssl_archive.parent.mkdir(parents=True)
        openssl_archive.touch()
        crypto_archive = openssl_archive.with_name("libcrypto.a")
        crypto_archive.touch()

        implicated = matrix.version_failure_libraries(
            "configure: error: OpenSSL version is too old and unsupported\n",
            ("glibc", "openssl"),
            {
                ("glibc", "libc.a"): glibc.archives[0],
                ("openssl", "libssl.a"): openssl_archive,
                ("openssl", "libcrypto.a"): crypto_archive,
            },
        )

        self.assertEqual(implicated, ["openssl"])

    def test_compile_error_for_library_api_identifies_library(self) -> None:
        glibc = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        acl_archive = self.root / "acl-2.3.2" / "libacl.a"
        acl_archive.parent.mkdir(parents=True)
        acl_archive.touch()
        log = (
            "set-permissions.c: In function 'set_acls':\n"
            "set-permissions.c:498:6: error: #error Must have acl_from_text\n"
        )

        implicated = matrix.version_failure_libraries(
            log,
            ("glibc", "acl"),
            {
                ("glibc", "libc.a"): glibc.archives[0],
                ("acl", "libacl.a"): acl_archive,
            },
        )

        self.assertEqual(implicated, ["acl"])

    def test_known_api_diagnostics_identify_owning_library(self) -> None:
        glibc = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        zlib_archive = self.root / "zlib-0.99" / "libz.a"
        zlib_archive.parent.mkdir(parents=True)
        zlib_archive.touch()
        overrides = {
            ("glibc", "libc.a"): glibc.archives[0],
            ("zlib", "libz.a"): zlib_archive,
        }

        cases = (
            (
                "cramfs_common.c:114: undefined reference to `zError'\n",
                ["zlib"],
            ),
            (
                "include/unistd.h:1208: error: static declaration of "
                "'close_range' follows non-static declaration\n",
                ["glibc"],
            ),
            (
                "pool.c:42: undefined reference to `pthread_mutex_lock'\n",
                ["glibc"],
            ),
            (
                "/usr/bin/ld: libgcc_eh.a(unwind-dw2-fde-dip.o): "
                "undefined reference to `_dl_find_object'\n",
                ["glibc"],
            ),
        )
        for log, expected in cases:
            with self.subTest(log=log):
                self.assertEqual(
                    matrix.version_failure_libraries(
                        log,
                        ("glibc", "zlib"),
                        overrides,
                    ),
                    expected,
                )

    def test_failure_log_includes_nested_config_log_tail(self) -> None:
        output_dir = self.root / "output"
        config_log = output_dir / "build" / "project" / "config.log"
        config_log.parent.mkdir(parents=True)
        config_log.write_text(
            "checking zstd... no\n"
            "configure: error: Failed to find ZSTD_minCLevel function\n"
        )

        log = matrix.build_failure_log(output_dir)

        self.assertIn(str(config_log), log)
        self.assertIn("ZSTD_minCLevel", log)

    def test_failure_log_includes_nested_glibc_abi_error(self) -> None:
        output_dir = self.root / "output"
        config_log = output_dir / "build" / "config.log"
        config_log.parent.mkdir(parents=True)
        config_log.write_text(
            "gcc conftest.c /tmp/glibc-2.17/lib/libc.a\n"
            "/usr/bin/ld: libgcc_eh.a(unwind-dw2-fde-dip.o): "
            "undefined reference to `_dl_find_object'\n"
            "configure: error: C compiler cannot create executables\n"
        )

        log = matrix.build_failure_log(output_dir)

        self.assertIn(str(config_log), log)
        self.assertIn("_dl_find_object", log)

    def test_failed_build_retries_with_next_glibc_version(self) -> None:
        initial = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        current = self.variant(
            "glibc-2.41", "current", "gcc-13", "Os"
        )
        minor = self.variant(
            "glibc-2.39", "minor-alternative", "gcc-13", "Os"
        )
        matrix.MATRIX_LIBRARY_VARIANTS.clear()
        matrix.MATRIX_LIBRARY_VARIANTS["glibc"] = [initial, current, minor]
        output_dir = self.root / "output"
        successful_case = {
            "variant": "bash_gcc-11-11.5.0_O0",
            "program": "bash",
            "binary": str(output_dir / "bash"),
        }
        error = subprocess.CalledProcessError(1, ["make"])
        log = (
            f"/usr/bin/ld: {initial.archives[0]}(malloc.o): "
            "multiple definition of `malloc'\n"
        )

        with (
            mock.patch.object(
                matrix,
                "case_output_dir",
                return_value=output_dir,
            ),
            mock.patch.object(
                matrix,
                "build_failure_log",
                return_value=log,
            ),
            mock.patch.object(
                matrix,
                "build_case",
                side_effect=[error, successful_case],
            ) as build,
        ):
            result = matrix.build_case_with_version_fallbacks(
                seed=7,
                program="bash",
                compiler="gcc-11",
                elf_optimization="O0",
                libraries=("glibc",),
                selected={"glibc": "Os"},
                archive_overrides={("glibc", "libc.a"): initial.archives[0]},
                forced={},
                jobs=2,
                clean=False,
                skip_existing=True,
                dry=False,
            )

        self.assertEqual(build.call_count, 2)
        second_overrides = build.call_args_list[1].args[8]
        self.assertEqual(
            second_overrides[("glibc", "libc.a")],
            current.archives[0],
        )
        self.assertTrue(result["library_version_fallback"]["used"])
        self.assertEqual(
            result["library_version_fallback"]["attempts"][-1][
                "changed_library"
            ],
            "glibc",
        )

    def test_fallback_runs_for_every_compiler_shard(self) -> None:
        initial = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        current = self.variant(
            "glibc-2.41", "current", "gcc-13", "Os"
        )
        matrix.MATRIX_LIBRARY_VARIANTS.clear()
        matrix.MATRIX_LIBRARY_VARIANTS["glibc"] = [initial, current]
        error = subprocess.CalledProcessError(1, ["make"])

        for compiler in ("gcc-11", "gcc-13", "clang-14", "clang-18"):
            with self.subTest(compiler=compiler):
                output_dir = self.root / compiler
                successful_case = {
                    "variant": f"bash_{compiler}_O0",
                    "program": "bash",
                    "binary": str(output_dir / "bash"),
                }
                with (
                    mock.patch.object(
                        matrix,
                        "case_output_dir",
                        return_value=output_dir,
                    ),
                    mock.patch.object(
                        matrix,
                        "build_failure_log",
                        return_value=(
                            "ld.lld: error: undefined symbol: "
                            "__isoc23_strtol\n"
                        ),
                    ),
                    mock.patch.object(
                        matrix,
                        "build_case",
                        side_effect=[error, successful_case],
                    ) as build,
                ):
                    result = matrix.build_case_with_version_fallbacks(
                        seed=7,
                        program="bash",
                        compiler=compiler,
                        elf_optimization="O0",
                        libraries=("glibc",),
                        selected={"glibc": "Os"},
                        archive_overrides={
                            ("glibc", "libc.a"): initial.archives[0]
                        },
                        forced={},
                        jobs=2,
                        clean=False,
                        skip_existing=True,
                        dry=False,
                    )

                self.assertEqual(build.call_count, 2)
                self.assertEqual(build.call_args_list[0].args[1], compiler)
                self.assertEqual(build.call_args_list[1].args[1], compiler)
                second_overrides = build.call_args_list[1].args[8]
                self.assertEqual(
                    second_overrides[("glibc", "libc.a")],
                    current.archives[0],
                )
                self.assertTrue(result["library_version_fallback"]["used"])

    def test_fallback_changes_multiple_libraries_cumulatively(self) -> None:
        glibc_old = self.variant(
            "glibc-2.17", "major-alternative", "gcc-13", "Os"
        )
        glibc_current = self.variant(
            "glibc-2.41", "current", "gcc-13", "Os"
        )
        zstd_old = self.variant(
            "zstd-0.8.1",
            "major-alternative",
            "gcc-13",
            "Os",
            "libzstd.a",
        )
        zstd_current = self.variant(
            "zstd-1.5.7",
            "current",
            "gcc-13",
            "Os",
            "libzstd.a",
        )
        matrix.MATRIX_LIBRARY_VARIANTS.clear()
        matrix.MATRIX_LIBRARY_VARIANTS.update({
            "glibc": [glibc_old, glibc_current],
            "zstd": [zstd_old, zstd_current],
        })
        output_dir = self.root / "output"
        successful_case = {
            "variant": "rsync_gcc-11-11.5.0_O2",
            "program": "rsync",
            "binary": str(output_dir / "rsync"),
        }
        error = subprocess.CalledProcessError(1, ["make"])

        with (
            mock.patch.object(
                matrix,
                "case_output_dir",
                return_value=output_dir,
            ),
            mock.patch.object(
                matrix,
                "build_failure_log",
                side_effect=[
                    "x.c: undefined reference to `__isoc23_strtol'\n",
                    "configure: error: Failed to find "
                    "ZSTD_minCLevel function in zstd lib\n",
                ],
            ),
            mock.patch.object(
                matrix,
                "build_case",
                side_effect=[error, error, successful_case],
            ) as build,
        ):
            result = matrix.build_case_with_version_fallbacks(
                seed=7,
                program="rsync",
                compiler="gcc-11",
                elf_optimization="O2",
                libraries=("glibc", "zstd"),
                selected={"glibc": "Os", "zstd": "Os"},
                archive_overrides={
                    ("glibc", "libc.a"): glibc_old.archives[0],
                    ("zstd", "libzstd.a"): zstd_old.archives[0],
                },
                forced={},
                jobs=2,
                clean=False,
                skip_existing=True,
                dry=False,
            )

        self.assertEqual(build.call_count, 3)
        final_overrides = build.call_args_list[2].args[8]
        self.assertEqual(
            final_overrides[("glibc", "libc.a")],
            glibc_current.archives[0],
        )
        self.assertEqual(
            final_overrides[("zstd", "libzstd.a")],
            zstd_current.archives[0],
        )
        self.assertEqual(
            result["library_version_fallback"]["strategy"],
            "diagnostic_cumulative_source_version_v2",
        )


if __name__ == "__main__":
    unittest.main()
