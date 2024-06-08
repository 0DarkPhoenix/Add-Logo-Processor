import json
import logging
import os
import sys
import time
from pathlib import Path
from tkinter.tix import MAIN

import psutil
import requests

if getattr(sys, "frozen", False):
    # If the application is run as a bundled executable, use the directory of the executable
    MAIN_PATH = os.path.dirname(sys.executable)
else:
    # Otherwise, just use the normal directory where the script resides
    MAIN_PATH = os.path.abspath(os.path.dirname(__file__))

CONFIG_PATH = Path(MAIN_PATH, "config.json")
LOG_FILE_PATH = Path(MAIN_PATH, "updater.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)


# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception


def get_downgrade_version(config: dict) -> str | None:
    """
    Get the version of the locally installed application to downgrade to

    :return: string with the version to downgrade to or None
    """
    return config.get("downgrade_version")

def get_latest_version(repo_url: str) -> str | None:
    """
    Get the latest version of the application from the GitHub repository

    :return: string with the latest version or None
    """
    response = requests.get(repo_url + "version.txt")

    if response.status_code != 200:
        return None

    return response.text.strip()

def download_and_install_version(release_url: str, filename: str) -> bool:
    """
    Download and install the given version of the application

    :param release_url: URL to the GitHub release
    :param filename: name of the file to download

    :return: True if the download and installation was successful, False otherwise
    """
    try:
        exe_url = f"{release_url}/{filename}"
        response = requests.get(exe_url, allow_redirects=True)

        if response.status_code != 200:
            logging.error(
                f"Failed to download {filename} from {release_url}. Status code: {response.status_code}. URL: {exe_url}"
            )
            return False

        # Kill the main executable
        kill_process(filename)
        time.sleep(1)

        # Overwrite the old executable with the new executable
        with open(filename, "wb") as file:
            file.write(response.content)
        logging.info(f"Successfully downloaded and installed {filename}")
        return True

    except requests.RequestException as e:
        logging.error(
            f"RequestException while trying to download {filename} from {release_url}: {e}"
        )
        return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while downloading and installing {filename}: {e}"
        )
        return False


def kill_process(process_name: str) -> None:
    """
    Kill all processes with the given name of the executable

    :param process_name: name of the executable to kill
    """
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == process_name:
            proc.kill()


def convert_json_files(release_url: str) -> None:
    """
    Convert the local JSON files to the new JSON files

    :param release_url: URL to the GitHub release
    """
    json_files = [file for file in os.listdir(MAIN_PATH) if file.endswith(".json")]

    for json_file in json_files:
        json_url = f"{release_url}/{json_file}"
        json_response = requests.get(json_url, allow_redirects=True)

        if json_response.status_code != 200:
            logging.error(
                f"Failed to download {json_file} from the repository. URL: {json_url}"
            )
            continue

        new_settings = json_response.json()
        json_filename = os.path.splitext(json_file)[0]

        local_file_path = os.path.join(MAIN_PATH, json_filename + ".json")

        try:
            with open(local_file_path, "r") as local_file:
                local_settings = json.load(local_file)
        except FileNotFoundError:
            logging.error(f"Local file {local_file_path} not found.")
            continue
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {local_file_path}.")
            continue

        updated_settings = {}
        for key, value in new_settings.items():
            if key in local_settings:
                updated_settings[key] = local_settings[key]
            else:
                updated_settings[key] = value

        try:
            with open(local_file_path, "w") as local_file:
                json.dump(updated_settings, local_file, indent=4)
            logging.info(f"Updated {json_file} successfully.")
        except Exception as e:
            logging.error(f"Error writing to {local_file_path}: {e}")


def load_config() -> dict:
    """
    Loads the config from config.json

    :return: dictionary with all the config items from config.json
    """
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    return config


def save_config(config: dict) -> None:
    """
    Saves the config to config.json

    :param config: dictionary with all the config items to save to config.json
    """

    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)


def main() -> None:
    logging.info(f"{'#'*40} Updater application has started {'#'*40}")

    exe_filename = "Add Logo Processor.exe"
    try:
        config = load_config()

        repo_url = config["repo_url"]

        version = (
            get_latest_version(repo_url)
            if get_downgrade_version(config) == ""
            else get_downgrade_version(config)
        )

        # main executable is always located in /Release/[version]
        releases_url = repo_url + "Release/" + version

        if download_and_install_version(releases_url, exe_filename):
            # Convert JSON files
            convert_json_files(releases_url)

            # Write version to config.json
            config["version"] = version

            # Turn downgrade_version back to an empty string
            config["downgrade_version"] = ""

            # Save the config
            save_config(config)

            logging.info("Update successful.")

        else:
            logging.error("Failed to download the new version.")

    except Exception as e:
        logging.error(f"Error: {e}")

    # Start the executable regardless of whether the download was successful or not
    os.startfile(str(Path(MAIN_PATH, exe_filename)))

if __name__ == "__main__":
    main()
