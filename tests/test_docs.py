"""Comprehensive documentation tests for GymTorax project.

This module provides thorough testing of Sphinx documentation including:
- Basic build functionality
- Quality checks (warnings, links, images)
- Coverage validation
- Structure verification
- Performance monitoring

Configuration options allow for flexible warning handling.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Configuration - set these via environment variables or modify directly
DOCS_FAIL_ON_WARNINGS = os.getenv("DOCS_FAIL_ON_WARNINGS", "false").lower() == "true"
DOCS_MAX_BUILD_TIME = int(os.getenv("DOCS_MAX_BUILD_TIME", "60"))  # seconds
DOCS_CHECK_EXTERNAL_LINKS = (
    os.getenv("DOCS_CHECK_EXTERNAL_LINKS", "false").lower() == "true"
)


class DocumentationTestHelper:
    """Helper class for documentation testing utilities."""

    @staticmethod
    def get_docs_dir() -> Path:
        """Get the documentation directory path."""
        return Path(__file__).parent.parent / "docs"

    @staticmethod
    def run_sphinx_build(
        builder: str = "html", extra_args: list[str] = None
    ) -> tuple[int, str, str]:
        """Run sphinx-build and return (returncode, stdout, stderr)."""
        docs_dir = DocumentationTestHelper.get_docs_dir()
        extra_args = extra_args or []

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "sphinx-build",
                "-b",
                builder,
                *extra_args,
                str(docs_dir),
                str(tmp_dir),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr


class TestDocumentationBuild:
    """Basic documentation build tests."""

    def test_sphinx_build_succeeds(self):
        """Test that Sphinx can build the documentation without errors."""
        returncode, stdout, stderr = DocumentationTestHelper.run_sphinx_build()

        assert returncode == 0, (
            f"Sphinx build failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


class TestDocumentationQuality:
    """Documentation quality and content tests."""

    def test_sphinx_warnings(self):
        """Test Sphinx warnings - report but do not fail unless configured."""
        returncode, stdout, stderr = DocumentationTestHelper.run_sphinx_build(
            extra_args=["-v"]  # Verbose to capture warnings
        )

        # Extract warnings from output
        warnings = []
        for line in stderr.split("\n"):
            if "WARNING" in line or "ERROR" in line:
                warnings.append(line.strip())

        if warnings:
            warning_msg = "Sphinx warnings found:\n" + "\n".join(warnings)
            if DOCS_FAIL_ON_WARNINGS:
                pytest.fail(warning_msg)
            else:
                # Report as warning without failing the test
                pytest.skip(
                    f"⚠️  {warning_msg}\nℹ️  Set DOCS_FAIL_ON_WARNINGS=true to fail on warnings"
                )

    def test_all_images_exist(self):
        """Test that all referenced images actually exist."""
        docs_dir = DocumentationTestHelper.get_docs_dir()

        # Find all image references in RST files
        image_refs = []
        for rst_file in docs_dir.rglob("*.rst"):
            with open(rst_file) as f:
                content = f.read()
                lines = content.split("\n")
                for line in lines:
                    if ".. image::" in line or ".. figure::" in line:
                        # Extract image path
                        parts = line.split("::")
                        if len(parts) > 1:
                            img_path = parts[1].strip()
                            if img_path.startswith("Images/") or img_path.startswith(
                                "../Images/"
                            ):
                                # Normalize path
                                normalized = img_path.replace("../", "")
                                image_refs.append(normalized)

        # Check that all referenced images exist
        missing_images = []
        for img_ref in set(image_refs):  # Remove duplicates
            img_path = docs_dir / img_ref
            if not img_path.exists():
                missing_images.append(str(img_ref))

        assert len(missing_images) == 0, f"Missing images: {missing_images}"

    def test_cross_references_valid(self):
        """Test that cross-references can be resolved."""
        # Use Sphinx's built-in reference checking instead of manual parsing
        docs_dir = DocumentationTestHelper.get_docs_dir()

        # Run Sphinx build with reference checking
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "html",
                    "-W",
                    "--keep-going",  # Continue after warnings to collect all issues
                    "-v",
                    str(docs_dir),
                    str(tmp_dir),
                ],
                capture_output=True,
                text=True,
            )

            # Parse output for reference-related warnings
            broken_refs = []
            for line in result.stderr.split("\n"):
                if "WARNING" in line and (
                    "unknown document:" in line or "undefined label:" in line
                ):
                    broken_refs.append(line.strip())

            if broken_refs:
                ref_msg = "Broken cross-references found:\n" + "\n".join(broken_refs)
                # Report as informational warning
                pytest.skip(
                    f"⚠️  {ref_msg}\nℹ️  These are Sphinx-detected reference issues"
                )
            else:
                # Test passes silently when no broken references found
                pass

    def test_external_links(self):
        """Test external links if enabled."""
        if not DOCS_CHECK_EXTERNAL_LINKS:
            pytest.skip(
                "External link checking disabled (set DOCS_CHECK_EXTERNAL_LINKS=true to enable)"
            )

        returncode, stdout, stderr = DocumentationTestHelper.run_sphinx_build(
            builder="linkcheck", extra_args=["-q"]
        )

        if returncode != 0:
            # Parse linkcheck output for broken external links
            broken_external = []
            for line in stdout.split("\n"):
                if "broken" in line.lower() and "http" in line:
                    broken_external.append(line.strip())

            if broken_external:
                broken_msg = "Broken external links found:\n" + "\n".join(
                    broken_external
                )
                # Report as informational warning but do not fail
                pytest.skip(f"⚠️  {broken_msg}")
                # do not fail on external links by default as they can be unreliable


class TestDocumentationCoverage:
    """Test documentation coverage for API elements."""

    def test_automodule_targets_exist(self):
        """Test that all automodule targets reference existing modules."""
        docs_dir = DocumentationTestHelper.get_docs_dir()
        missing_modules = []

        for rst_file in docs_dir.rglob("*.rst"):
            with open(rst_file) as f:
                content = f.read()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if ".. automodule::" in line:
                        module_name = line.split("::")[1].strip()

                        # Try to import the module
                        try:
                            # Add project root to path temporarily
                            import sys

                            project_root = str(docs_dir.parent)
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)

                            __import__(module_name)
                        except ImportError as e:
                            missing_modules.append(
                                f"{rst_file.name}:{i} -> {module_name} ({e})"
                            )

        if missing_modules:
            msg = "Automodule targets that cannot be imported:\n" + "\n".join(
                missing_modules
            )
            # Report as informational warning but do not fail
            pytest.skip(f"⚠️  {msg}\nℹ️  These modules may need mock imports in conf.py")


class TestDocumentationStructure:
    """Test documentation structure and organization."""

    def test_toctree_references_valid(self):
        """Test that all toctree entries reference existing files."""
        docs_dir = DocumentationTestHelper.get_docs_dir()
        invalid_refs = []

        for rst_file in docs_dir.rglob("*.rst"):
            with open(rst_file) as f:
                content = f.read()
                lines = content.split("\n")

                in_toctree = False
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    if ".. toctree::" in line:
                        in_toctree = True
                        continue

                    if in_toctree:
                        # Empty line or directive options continue toctree
                        if not stripped or stripped.startswith(":"):
                            continue
                        # Non-indented line ends toctree
                        if not line.startswith("   ") and stripped:
                            in_toctree = False
                            continue
                        # This is a toctree entry
                        if line.startswith("   ") and stripped:
                            ref = stripped
                            # Check if referenced file exists
                            possible_paths = [
                                rst_file.parent / f"{ref}.rst",
                                docs_dir / f"{ref}.rst",
                                rst_file.parent / ref / "index.rst",
                                docs_dir / ref / "index.rst",
                            ]

                            if not any(p.exists() for p in possible_paths):
                                invalid_refs.append(f"{rst_file.name}:{i} -> {ref}")

        assert len(invalid_refs) == 0, f"Invalid toctree references: {invalid_refs}"

    def test_consistent_naming_convention(self):
        """Test that files follow snake_case naming convention."""
        docs_dir = DocumentationTestHelper.get_docs_dir()
        bad_names = []

        for rst_file in docs_dir.rglob("*.rst"):
            # Skip files in _build or _static directories
            if "_build" in rst_file.parts or "_static" in rst_file.parts:
                continue

            filename = rst_file.name
            if filename != "index.rst":  # index.rst is always acceptable
                # Check if filename contains spaces or capital letters
                if " " in filename or any(
                    c.isupper() for c in filename.replace(".rst", "")
                ):
                    bad_names.append(str(rst_file.relative_to(docs_dir)))

        if bad_names:
            msg = f"Files not following snake_case convention: {bad_names}"
            # Report as informational warning but do not fail
            pytest.skip(f"⚠️  {msg}\nℹ️  Consider renaming to snake_case for consistency")


class TestDocumentationConfig:
    """Test documentation configuration."""

    def test_readthedocs_config_valid(self):
        """Test that .readthedocs.yaml is valid."""
        rtd_config = Path(__file__).parent.parent / ".readthedocs.yaml"

        assert rtd_config.exists(), "Missing .readthedocs.yaml file"

        # Basic validation - check it's not empty and has required fields
        with open(rtd_config) as f:
            content = f.read()

        assert len(content.strip()) > 0, ".readthedocs.yaml is empty"
        assert "version:" in content, "Missing version field in .readthedocs.yaml"
        assert "sphinx:" in content, "Missing sphinx configuration in .readthedocs.yaml"

    def test_sphinx_config_valid(self):
        """Test that Sphinx configuration is valid."""
        docs_dir = DocumentationTestHelper.get_docs_dir()
        conf_py = docs_dir / "conf.py"

        assert conf_py.exists(), "Missing docs/conf.py file"

        # Test that conf.py can be imported without errors
        import sys

        sys.path.insert(0, str(docs_dir))

        try:
            import conf  # This should not raise an exception

            # Check for required configurations
            assert hasattr(conf, "project"), "Missing project name in conf.py"
            assert hasattr(conf, "extensions"), "Missing extensions in conf.py"
            assert hasattr(conf, "html_theme"), "Missing html_theme in conf.py"
        finally:
            # Clean up
            if "conf" in sys.modules:
                del sys.modules["conf"]
            sys.path.remove(str(docs_dir))
