"""Test package bootstrap."""

from tests.isolation import TEST_OUTPUTS_ROOT, install_test_output_isolation

TEST_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
install_test_output_isolation()
