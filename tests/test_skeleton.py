"""Skeleton sanity tests — verify that the package imports cleanly."""


def test_import_stratego() -> None:
    """Top-level package must import without error."""
    import stratego

    assert stratego.__version__ == "0.1.0"


def test_import_subpackages() -> None:
    """Every subpackage must be importable from a fresh interpreter."""
    import stratego.env  # noqa: F401
    import stratego.networks  # noqa: F401
    import stratego.search  # noqa: F401
    import stratego.training  # noqa: F401
    import stratego.utils  # noqa: F401


def test_conftest_device_fixture() -> None:
    """The device fixture should resolve to either cuda or cpu."""
    import torch

    assert torch.device("cuda" if torch.cuda.is_available() else "cpu").type in ("cuda", "cpu")
