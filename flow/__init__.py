"""codicodec-flow: a generative flow-matching model on CoDiCodec latents."""

# ---------------------------------------------------------------------------
# When running from the repo root, the cwd `./codicodec/` directory shadows
# the editable-installed `codicodec` package via Python's namespace-package
# discovery. setuptools registers an `_EditableFinder` in sys.meta_path that
# *would* resolve the import correctly, but it is appended after the default
# `PathFinder`, so it is never reached. We move it to the front here, before
# any other module in this package imports `codicodec`.
# ---------------------------------------------------------------------------
def _prioritize_editable_finders() -> None:
    import sys
    for finder in list(sys.meta_path):
        name = getattr(finder, "__name__", "") or type(finder).__name__
        if "Editable" in name:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                continue
            sys.meta_path.insert(0, finder)


_prioritize_editable_finders()
del _prioritize_editable_finders

__version__ = "0.0.1"
