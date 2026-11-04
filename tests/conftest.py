import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def disable_interactive_plots(monkeypatch):
    """
    Ensure all tests run completely headless and no plots emerge.
    """
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
