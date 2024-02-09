from ... import DLProblem
from pathlib import Path
import platform


def get_path() -> Path:
    """Get the path to the problem DLL/SO file."""
    ext = ".dll" if platform.system() == "Windows" else ".so"
    fname = "alpaqa-lasso" + ext
    return Path(__file__).parent / fname


def load(*args, **kwargs):
    """Load the problem file as an alpaqa problem."""
    return DLProblem(str(get_path()), *args, **kwargs)
