import os

import psutil
import requests
from numpy import delete

# TODO: Add code which always converts the old settings file with the new settings file using templates and filling it in with the settings it currently has


def get_current_version():
    try:
        with open("version.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def get_downgrade_version():
    try:
        with open("downgrade_version.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return False


def get_latest_version(repo_url):
    response = requests.get(repo_url + "version.txt")
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None


def download_version(repo_url, filename):
    response = requests.get(repo_url + filename, allow_redirects=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        return True
    return False


def kill_process(process_name):
    """Kill all processes with the given name."""
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == process_name:
            proc.kill()


def main():
    repo_url = "https://raw.githubusercontent.com/0DarkPhoenix/Add-Logo-Processor/main/"

    version = (
        get_latest_version(repo_url)
        if not get_downgrade_version()
        else get_downgrade_version()
    )

    # main executable is always located in /Release/[version]
    releases_url = repo_url + "Release" + version

    exe_filename = "Add Logo Processor.exe"

    kill_process(exe_filename)

    if download_version(releases_url, exe_filename):

        # Write version to version.txt
        with open("version.txt", "w") as file:
            file.write(version)

        # If downgrade.txt exists, delete it
        try:
            os.remove("downgrade_version.txt")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error removing downgrade_version.txt: {e}")

        print("Update successful.")
    else:
        print("Failed to download the new version.")


if __name__ == "__main__":
    main()
