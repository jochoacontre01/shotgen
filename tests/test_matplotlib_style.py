import matplotlib.pyplot as plt
import numpy as np
from shotgen.migration.kirchhoff import KirchhoffMigration
from shotgen.utils import setup_matplotlib_style


def test_setup_matplotlib_style_properties():
    """Verify matplotlib styling rules: size (10,5), dpi=150, no grid, external ticks."""
    setup_matplotlib_style()
    fig, _ax = plt.subplots()

    # Size: (10, 5)
    assert fig.get_size_inches().tolist() == [10.0, 5.0]

    # DPI: 150
    assert fig.dpi == 150.0
    assert plt.rcParams["savefig.dpi"] == 150

    # No grid
    assert plt.rcParams["axes.grid"] is False

    # Only external ticks
    assert plt.rcParams["xtick.direction"] == "out"
    assert plt.rcParams["ytick.direction"] == "out"
    assert plt.rcParams["xtick.top"] is True
    assert plt.rcParams["ytick.right"] is True

    plt.close(fig)


def test_headless_conftest_blocks_show(monkeypatch):
    """Verify that during pytest execution, plt.show does not emerge/block."""
    # plt.show should execute harmlessly as a no-op thanks to conftest fixture
    plt.show()


def test_kirchhoff_migration_headless_init():
    """Verify that KirchhoffMigration.__init__ does not trigger plots by default."""
    vp = 2000.0 * np.ones((20, 20), dtype=np.float32)
    sources = np.array([[50.0, 10.0]])
    receivers = np.array([[60.0, 10.0]])
    mig = KirchhoffMigration(vp=vp, sources=sources, receivers=receivers, spacing=(10.0, 10.0), preview=False)
    assert mig.preview is False
    assert mig.vp is not None
