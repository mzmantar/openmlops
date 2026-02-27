import subprocess

def dvc_pull() -> None:
    # Works even if no remote: ensures workspace matches DVC metadata
    subprocess.run(["dvc", "pull"], check=False)