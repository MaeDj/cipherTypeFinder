
###Claude suggested helper###
import os
from pathlib import Path

import paramiko
from dotenv import load_dotenv

# Shared .env at the project root, holding SFTP_HOST/PORT/USERNAME/PASSWORD.
load_dotenv(Path(__file__).resolve().parents[2] / '.env')

_ssh = None
_sftp = None


def _get_sftp():
    global _ssh, _sftp
    if _sftp is None:
        #Reached only when the requested path doesn't exist locally (see listdir below), so a
        #missing var here means SFTP genuinely wasn't configured for this host - surface that
        #plainly instead of a bare "KeyError: 'SFTP_HOST'" with no indication of what to fix.
        missing = [var for var in ('SFTP_HOST', 'SFTP_USERNAME') if var not in os.environ]
        if missing:
            raise RuntimeError(
                f"Path not found locally and SFTP isn't configured: missing {', '.join(missing)} "
                f"in the environment/.env. Set them, or run on the host that holds the data."
            )
        _ssh = paramiko.SSHClient()
        _ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _ssh.connect(
            hostname=os.environ['SFTP_HOST'],
            port=int(os.environ.get('SFTP_PORT', 22)),
            username=os.environ['SFTP_USERNAME'],
            password=os.environ.get('SFTP_PASSWORD'),
        )
        _sftp = _ssh.open_sftp()
    return _sftp


def listdir(remote_path):
    """Drop-in replacement for os.listdir().

    Uses the local filesystem when remote_path already exists locally
    (e.g. running directly on the host that holds the corpus), and only
    falls back to SFTP when it doesn't - so no SFTP_* env vars are needed
    when running on the data host itself.
    """
    local_path = os.path.expanduser(remote_path)
    if os.path.isdir(local_path):
        return os.listdir(local_path)
    return _get_sftp().listdir(remote_path)


def close():
    global _ssh, _sftp
    if _sftp is not None:
        _sftp.close()
        _sftp = None
    if _ssh is not None:
        _ssh.close()
        _ssh = None