"""Repository-root pytest configuration: registers the test-lane collection policy plugin.

``pytest_plugins`` is honored only in the root conftest, which is why the registration lives here
rather than in ``tests/conftest.py`` (audit 2026-10-08 P2-8).
"""
pytest_plugins = ["tests.pytest_policy"]
