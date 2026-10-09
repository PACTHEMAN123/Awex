"""Weight communication with one CUDA transport and reusable plans."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "Transport":
        from .transport import Transport

        return Transport
    raise AttributeError(name)
