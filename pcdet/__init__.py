import subprocess
from pathlib import Path

from .version import __version__

__all__ = [
    '__version__'
]


def get_git_commit_number():
    if not (Path(__file__).parent / '../.git').exists():
        return '0000000'

    try:
        cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, timeout=1)
        if cmd_out.returncode == 0:
            git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
            return git_commit_number
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Git is not available or command timed out (e.g., in Docker without git)
        pass
    
    return '0000000'


script_version = get_git_commit_number()


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
