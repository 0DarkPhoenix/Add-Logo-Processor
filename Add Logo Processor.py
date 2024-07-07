import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import aiohttp
import customtkinter as ctk
import cv2
import numpy as np
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip
from packaging import version
from PIL import Image

ctk.set_appearance_mode("dark")

global main_window  # main_window gets defined if __name__ == "__main__"

if getattr(sys, "frozen", False):
    # If the application is run as a bundled executable, use the directory of the executable
    MAIN_PATH = os.path.dirname(sys.executable)
else:
    # Otherwise, just use the normal directory where the script resides
    MAIN_PATH = os.path.abspath(os.path.dirname(__file__))

SETTINGS_PATH = Path(MAIN_PATH, "settings.json")
CONFIG_PATH = Path(MAIN_PATH, "config.json")
LOG_FILE_PATH = Path(MAIN_PATH, "add_logo_processor.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)


# Global exception handler
def handle_exception(exc_type: type, exc_value: Exception, exc_traceback: traceback):
    """Handle uncaught exceptions by logging them.

    Args:
        exc_type (type): Exception type.
        exc_value (Exception): Exception instance.
        exc_traceback (traceback): Traceback object.

    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# Colors
DEFAULT_WHITE = "#dce4ee"
DEFAULT_GRAY = "#737373"

# Constants
SCALE_DEFAULT = 10
OFFSET_DEFAULT = 10
LOGO_CORNER_DEFAULT = "Bottom Left"
IMAGE_FORMATS = ["png", "jpg", "webp"]
VIDEO_SCALE_DEFAULT = 13
VIDEO_WIDTH_OFFSET_DEFAULT = 10
VIDEO_HEIGHT_OFFSET_DEFAULT = 15

# ----------------------------------- v1.5 ----------------------------------- #
# TODO: Write code for when the user wants to downgrade their current version of the application


class UpdateAvailableWindow(ctk.CTk):
    def __init__(self):
        """Initialize the UpdateAvailableWindow class."""
        super().__init__()

    def run(self, current_version: float, latest_version: float):
        """Run the update available window.

        Args:
            current_version (float): The current version of the application.
            latest_version (float): The latest version available for the application.

        """
        self.title("Update Available")
        self.geometry("400x200")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.create_window_elements(current_version, latest_version)

    def create_window_elements(self, current_version: float, latest_version: float):
        """Create the elements for the update available window.

        Args:
            current_version (float): The current version of the application.
            latest_version (float): The latest version available for the application.

        """
        label_title = ctk.CTkLabel(self, text="A new version is available!", font=("Arial", 22))
        label_title.place(relx=0.5, rely=0.2, anchor="center")

        label_versions = ctk.CTkLabel(
            self,
            text=f"{current_version} (Current)  →  {latest_version}",
            font=("Arial", 14),
        )
        label_versions.place(relx=0.5, rely=0.4, anchor="center")

        label_update_question = ctk.CTkLabel(
            self, text="Would you like to update?", font=("Arial", 14)
        )
        label_update_question.place(relx=0.5, rely=0.6, anchor="center")

        button_yes = ctk.CTkButton(self, text="Yes", command=self.update, font=("Arial", 14))
        button_yes.place(relx=0.3, rely=0.8, anchor="center")

        button_no = ctk.CTkButton(self, text="No", command=self.close, font=("Arial", 14))
        button_no.place(relx=0.7, rely=0.8, anchor="center")

    def update(self):
        """Update the application by running the updater executable."""
        updater_path = Path(MAIN_PATH, "Updater.exe")
        subprocess.Popen([str(updater_path)], shell=True)
        os._exit(0)

    def close(self):
        """Close the update available window."""
        self.destroy()


class ImageProcessor:
    @staticmethod
    def save_image_as_webp(output_file: str, img: np.ndarray, quality: Optional[int] = 80):
        """Save an image as a WebP file.

        Args:
            output_file (str): The path to the output file.
            img (np.ndarray): The image to be saved.
            quality (int, optional): The quality of the WebP image. Defaults to 80.

        """
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image.save(output_file, format="webp", quality=quality)

    @staticmethod
    def add_logo_operation(
        scale: int,
        width: int,
        height: int,
        new_height: int,
        new_width: int,
        logo_img: np.ndarray,
        img: np.ndarray,
        selected_corner: str,
        offset_logo_width: int,
        offset_logo_height: int,
    ):
        """Add a logo to an image.

        Args:
            scale (int): The scale of the logo.
            width (int): The width of the image.
            height (int): The height of the image.
            new_height (int): The new height of the image.
            new_width (int): The new width of the image.
            logo_img (np.ndarray): The logo image.
            img (np.ndarray): The image to which the logo will be added.
            selected_corner (str): The corner where the logo will be placed.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.

        """
        logo_height, logo_width, _ = logo_img.shape
        logo_aspectratio = logo_width / logo_height
        scale_factor = scale / 100

        if width < height:
            logo_height = new_height * scale_factor
            logo_width = logo_height * logo_aspectratio if logo_aspectratio != 1 else logo_height
        else:
            logo_width = new_width * scale_factor
            logo_height = logo_width / logo_aspectratio if logo_aspectratio != 1 else logo_width

        logo_img = cv2.resize(logo_img, (int(logo_width), int(logo_height)))

        offset_logo_width = (offset_logo_width / 100) * (
            new_width * 0.5
        )  # 0.5 is a reducing factor to allow more accurate control over offset_logo_width
        offset_logo_height = (offset_logo_height / 100) * (
            new_height * 0.5
        )  # 0.5 is a reducing factor to allow more accurate control over offset_logo_height

        match selected_corner:
            case "Top Left":
                position = int(0 + offset_logo_width), int(0 + offset_logo_height)
            case "Top Right":
                position = (
                    int(new_width - logo_width - offset_logo_width),
                    int(0 + offset_logo_height),
                )
            case "Bottom Left":
                position = (
                    int(0 + offset_logo_width),
                    int(new_height - logo_height - offset_logo_height),
                )
            case "Bottom Right":
                position = (
                    int(new_width - logo_width - offset_logo_width),
                    int(new_height - logo_height - offset_logo_height),
                )

        x_offset, y_offset = position
        y1, y2 = y_offset, y_offset + logo_img.shape[0]
        x1, x2 = x_offset, x_offset + logo_img.shape[1]

        alpha_s = logo_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = alpha_s * logo_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c]

    @staticmethod
    def resize_image_operation(
        input_folder_path: Path,
        output_folder_path: Path,
        file: str,
        min_pixels: int,
        overwrite_existing_images: bool,
        add_logo: bool,
        logo_img: np.ndarray,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
        convert_to_format: bool,
        format: str,
    ):
        """Resizes an image and optionally adds a logo.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            file (str): The file name of the image.
            min_pixels (int): The minimum number of pixels for the smallest side.
            overwrite_existing_images (bool): Whether to overwrite existing images.
            add_logo (bool): Whether to add a logo to the image.
            logo_img (np.ndarray): The logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.
            convert_to_format (bool): Whether to convert the image to a different format.
            format (str): The format to which the image will be converted.

        """
        output_file = Path(output_folder_path, file)

        if convert_to_format:
            output_file = Path(output_folder_path, file).with_suffix(f".{format}")

        if not output_file.exists() or overwrite_existing_images:
            # Open the input image
            input_file = Path(input_folder_path, file)
            img = cv2.imread(str(input_file))
            height, width, _ = img.shape

            # Calculate the new dimensions for resizing
            if width < height:
                new_width = min_pixels
                new_height = int(height * (min_pixels / width))
            else:
                new_height = min_pixels
                new_width = int(width * (min_pixels / height))
            img = cv2.resize(img, (new_width, new_height))

            if add_logo:
                ImageProcessor.add_logo_operation(
                    scale=scale,
                    width=width,
                    height=height,
                    new_height=new_height,
                    new_width=new_width,
                    logo_img=logo_img,
                    img=img,
                    selected_corner=selected_corner,
                    offset_logo_width=offset_logo_width,
                    offset_logo_height=offset_logo_height,
                )

            if format == "webp":
                ImageProcessor.save_image_as_webp(str(output_file), img)
            else:
                cv2.imwrite(str(output_file), img)

    @staticmethod
    def resize_images(
        input_folder_path: Path,
        output_folder_path: Path,
        min_pixels: int,
        overwrite_existing_images: bool,
        add_logo: bool,
        logo_path: Path,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
        convert_to_format: bool,
        format: str,
        clear_input_folder: bool,
        clear_output_folder: bool,
    ):
        """Resizes multiple images and optionally adds a logo.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            min_pixels (int): The minimum number of pixels for the smallest side.
            overwrite_existing_images (bool): Whether to overwrite existing images.
            add_logo (bool): Whether to add a logo to the images.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.
            convert_to_format (bool): Whether to convert the images to a different format.
            format (str): The format to which the images will be converted.
            clear_input_folder (bool): Whether to clear the input folder.
            clear_output_folder (bool): Whether to clear the output folder.

        """
        start_time = time.time()

        # Check if the min_pixels value hasn't changed since the last resize operation, otherwise it will overwrite all
        # images in the output folder
        try:
            settings = MainWindow.load_settings()
            settings_image_processor = settings["image_processor"]

            if settings_image_processor["number_pixels"] != min_pixels:
                overwrite_existing_images = True

            files = os.listdir(input_folder_path)
            files = sorted(
                files,
                key=lambda f: os.path.getsize(Path(input_folder_path, f)),
                reverse=True,
            )

            logo_img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)

            # Save all values to settings.json
            settings_image_processor.update(
                {
                    "input_folder_path": str(input_folder_path),
                    "output_folder_path": str(output_folder_path),
                    "number_pixels": min_pixels,
                    "add_logo": add_logo,
                    "logo_image_path": str(logo_path),
                    "scale": scale,
                    "width_offset": offset_logo_width,
                    "height_offset": offset_logo_height,
                    "logo_corner": selected_corner,
                    "convert_to_format": convert_to_format,
                    "format": format,
                    "clear_input_folder": clear_input_folder,
                    "clear_output_folder": clear_output_folder,
                }
            )
            MainWindow.save_settings(settings)
        except Exception as e:
            logging.error(f"Error: {e}")

        # Clear output folder if true
        if clear_output_folder:
            MainWindow.clear_folder(output_folder_path)

        # Create a progress bar
        total_files = len(files)
        MainWindow.create_progress_bar_image_processor(main_window, total_files)

        # Resize images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    ImageProcessor.resize_image_operation,
                    input_folder_path=input_folder_path,
                    output_folder_path=output_folder_path,
                    file=file,
                    min_pixels=min_pixels,
                    overwrite_existing_images=overwrite_existing_images,
                    add_logo=add_logo,
                    logo_img=logo_img,
                    scale=scale,
                    offset_logo_width=offset_logo_width,
                    offset_logo_height=offset_logo_height,
                    selected_corner=selected_corner,
                    convert_to_format=convert_to_format,
                    format=format,
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures), start=1):
                future.result()  # Wait for the resize_image operation to complete
                MainWindow.update_progress_bar_image_processor(main_window, total_files, i)

        # Clear input folder if true
        if clear_input_folder:
            MainWindow.clear_folder(input_folder_path)

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        logging.info(f"All images were processed in {execution_time} seconds")
        MainWindow.finish_progress_bar_image_processor(main_window, execution_time)


class VideoProcessor:
    @staticmethod
    def add_logo_to_video_operation(
        input_folder_path: Path,
        output_folder_path: Path,
        file,
        overwrite_existing_videos: bool,
        logo_path: Path,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        """Add a logo to a video.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            file (os.DirEntry): The video file.
            overwrite_existing_videos (bool): Whether to overwrite existing videos.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.

        """
        output_file = output_folder_path / file.name

        if not output_file.exists() or overwrite_existing_videos:
            # Load video clip
            video_path = str(input_folder_path / file.name)
            video_clip = VideoFileClip(video_path)

            # Load the logo image and resize it to fit the video dimensions
            logo_clip = ImageClip(str(logo_path))
            scale_factor = scale / 100

            if video_clip.h < video_clip.w:
                logo_clip = logo_clip.resize(height=int(video_clip.h * scale_factor))
            else:
                logo_clip = logo_clip.resize(height=int(video_clip.w * scale_factor))

            # Set the duration for both video clip and logo clip
            video_clip = video_clip.set_duration(video_clip.duration)
            logo_clip = logo_clip.set_duration(video_clip.duration)

            # Calculate the position to place the logo
            offset_logo_width = (offset_logo_width / 100) * logo_clip.w
            offset_logo_height = (offset_logo_height / 100) * logo_clip.h

            match selected_corner:
                case "Top Left":
                    position = (int(offset_logo_width), int(offset_logo_height))
                case "Top Right":
                    position = (
                        int(video_clip.w - logo_clip.w - offset_logo_width),
                        int(offset_logo_height),
                    )
                case "Bottom Left":
                    position = (
                        int(offset_logo_width),
                        int(video_clip.h - logo_clip.h - offset_logo_height),
                    )
                case "Bottom Right":
                    position = (
                        int(video_clip.w - logo_clip.w - offset_logo_width),
                        int(video_clip.h - logo_clip.h - offset_logo_height),
                    )

            # Overlay the logo on the video clip
            video_clip = CompositeVideoClip([video_clip, logo_clip.set_position(position)])

            # Set the output path and save the modified video clip
            video_clip.write_videofile(str(output_file), codec="libx264")

    @staticmethod
    def add_logo_to_video(
        input_folder_path: Path,
        output_folder_path: Path,
        overwrite_existing_videos: bool,
        logo_path: Path,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
        clear_input_folder: bool,
        clear_output_folder: bool,
    ):
        """Add a logo to multiple videos.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            overwrite_existing_videos (bool): Whether to overwrite existing videos.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.
            clear_input_folder (bool): Whether to clear the input folder.
            clear_output_folder (bool): Whether to clear the output folder.

        """
        start_time = time.time()

        files = sorted(os.scandir(input_folder_path), key=lambda f: f.stat().st_size, reverse=True)
        try:
            settings = MainWindow.load_settings()
            settings_video_processor = settings["video_processor"]
            settings_video_processor.update(
                {
                    "input_folder_path": str(input_folder_path),
                    "output_folder_path": str(output_folder_path),
                    "logo_image_path": str(logo_path),
                    "scale": scale,
                    "width_offset": offset_logo_width,
                    "height_offset": offset_logo_height,
                    "logo_corner": selected_corner,
                }
            )
            MainWindow.save_settings(settings)
        except Exception as e:
            logging.error(f"Error writing settings: {e}")

        # Clear output folder if true
        if clear_output_folder:
            MainWindow.clear_folder(output_folder_path)

        # Display Progress Bar
        total_files = len(files)
        MainWindow.create_progress_bar_video_processor(main_window, total_files)

        # Add logo to video in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    VideoProcessor.add_logo_to_video_operation,
                    input_folder_path=input_folder_path,
                    output_folder_path=output_folder_path,
                    file=file,
                    overwrite_existing_videos=overwrite_existing_videos,
                    logo_path=logo_path,
                    scale=scale,
                    offset_logo_width=offset_logo_width,
                    offset_logo_height=offset_logo_height,
                    selected_corner=selected_corner,
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures), start=1):
                future.result()  # Wait for the add_logo_to_video_operation to complete
                MainWindow.update_progress_bar_video_processor(main_window, total_files, i)

        # Clear input folder if true
        if clear_input_folder:
            MainWindow.clear_folder(input_folder_path)

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        logging.info(f"Processed all videos in {execution_time} seconds")
        MainWindow.finish_progress_bar_video_processor(main_window, execution_time)


class MainWindow(ctk.CTk):
    def __init__(self):
        """Initialize the MainWindow class."""
        super().__init__()

    def run(self):
        """Run the main window of the application."""
        self.title("Image & Video Processor")
        self.geometry("1000x1000")

        self.validate_numeric_input_cmd_command = (self.register(self.validate_numeric_input), "%P")
        self.setup_ui()

        self.insert_settings()

    def setup_ui(self):
        """Create the UI elements for the images and videos tab."""
        self.create_main_title()
        tabview = self.create_tabview()
        self.images_tab = tabview.add("Images")
        self.videos_tab = tabview.add("Videos")
        self.setup_image_tab()
        self.setup_video_tab()

    def create_main_title(self):
        """Create the main title."""
        main_title = ctk.CTkLabel(self, text="Image & Video Processor", font=("Arial", 28))
        main_title.place(relx=0.5, rely=0.03, anchor="center")

    def create_tabview(self):
        """Create the tab view.

        Returns
            ctk.CTkTabview: The created tab view.

        """
        tabview = ctk.CTkTabview(self, fg_color="#242424")
        tabview.place(relx=0, rely=0.07, relwidth=1, relheight=0.93)
        return tabview

    def setup_image_tab(self):
        """Create the image tab."""
        self.configure_grid(tab=self.images_tab, columns=3, rows=24)
        self.create_image_tab_elements()

    def setup_video_tab(self):
        """Create the video tab."""
        self.configure_grid(tab=self.videos_tab, columns=3, rows=22)
        self.create_video_tab_elements()

    def configure_grid(self, tab: ctk.CTkTabview, columns: int, rows: int):
        """Configure the grid layout for a tab.

        Args:
            tab (ctk.CTkTabview): The tab to configure.
            columns (int): Number of columns.
            rows (int): Number of rows.

        """
        for col in range(columns):
            tab.columnconfigure(col, weight=1)
        for row in range(rows):
            tab.rowconfigure(row, weight=1)

    def create_image_tab_elements(self):
        """Create elements for the image tab."""
        self.create_folder_input(self.images_tab, "Input Folder", 1, 2, 3)
        self.create_folder_input(self.images_tab, "Output Folder", 4, 5, 6)
        self.create_num_pixels_input()
        self.create_logo_image_elements()
        self.create_convert_format_elements()
        self.create_overwrite_checkbox(self.images_tab, 18)
        self.create_resize_button()

    def create_video_tab_elements(self):
        """Create elements for the video tab."""
        self.create_folder_input(self.videos_tab, "Input Folder", 1, 2, 3)
        self.create_folder_input(self.videos_tab, "Output Folder", 4, 5, 6)
        self.create_logo_video_elements()
        self.create_overwrite_checkbox(self.videos_tab, 15)
        self.create_add_logo_button()

    def create_folder_input(
        self, tab: ctk.CTkTabview, label_text: str, label_row: int, entry_row: int, button_row: int
    ):
        """Create folder input elements.

        Args:
            tab (ctk.CTkTabview): The tab to add the elements to.
            label_text (str): The text for the label.
            label_row (int): The row for the label.
            entry_row (int): The row for the entry.
            button_row (int): The row for the button.

        """
        label = ctk.CTkLabel(tab, text=label_text, font=("Arial", 14))
        label.grid(row=label_row, column=1, columnspan=3, sticky="s")

        entry = ctk.CTkEntry(tab, placeholder_text=f"Path of {label_text.lower()}", width=850)
        entry.grid(row=entry_row, column=1, columnspan=3)

        button = ctk.CTkButton(
            tab, text=f"Browse {label_text.lower()}", command=lambda: self.browse_path(entry)
        )
        button.grid(row=button_row, column=1, columnspan=3, sticky="n")

        if label_text == "Input Folder":
            if tab == self.images_tab:
                self.entry_input_image = entry
            else:
                self.entry_input_video = entry
        else:
            if tab == self.images_tab:
                self.entry_output_image = entry
            else:
                self.entry_output_video = entry

    def create_num_pixels_input(self):
        """Create the number of pixels input."""
        label = ctk.CTkLabel(
            self.images_tab,
            text="Maximum number of pixels of the smallest side",
            font=("Arial", 14),
        )
        label.grid(row=7, column=1, columnspan=3, sticky="s")

        self.entry_num_pixels = ctk.CTkEntry(
            self.images_tab,
            placeholder_text="Number of pixels",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_num_pixels.grid(row=8, column=1, columnspan=3)

    def create_logo_image_elements(self):
        """Create elements for the logo image."""
        self.frame_logo_image = ctk.CTkFrame(self.images_tab, fg_color="transparent")
        self.frame_logo_image.grid(row=10, rowspan=6, column=1, columnspan=3, sticky="nsew")

        self.toggle_logo_image = ctk.CTkSwitch(
            self.images_tab,
            text="Add Logo to Image",
            font=("Arial", 18),
            command=self.toggle_logo_actions,
        )
        self.toggle_logo_image.grid(row=10, column=1, columnspan=3)

        self.create_logo_image_path_elements()
        self.create_logo_image_scale_elements()
        self.create_logo_image_corner_selection()
        self.create_logo_image_offset_elements()

    def create_logo_image_path_elements(self):
        """Create elements for the logo image path."""
        self.label_logo_image = ctk.CTkLabel(
            self.images_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        self.label_logo_image.grid(row=11, column=2, sticky="s")

        self.entry_logo_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Path to Logo Image", width=600
        )
        self.entry_logo_image.grid(row=12, column=2)

        self.browse_button_logo_image = ctk.CTkButton(
            self.images_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_image),
        )
        self.browse_button_logo_image.grid(row=13, column=2, sticky="n")

    def create_logo_image_scale_elements(self):
        """Create elements for the logo image scale."""
        self.label_scale_logo_image = ctk.CTkLabel(
            self.images_tab,
            text=f"Scale of the logo image [Range: 1 to 100 , Default: {SCALE_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_scale_logo_image.grid(row=14, column=2, sticky="s")

        self.entry_scale_logo_image = ctk.CTkEntry(
            self.images_tab,
            placeholder_text="Scale",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_scale_logo_image.grid(row=15, column=2, sticky="n")

    def create_logo_image_corner_selection(self):
        """Create elements for the logo image corner selection."""
        self.label_corner_selection_image = ctk.CTkLabel(
            self.images_tab, text="Corner Selection", font=("Arial", 14)
        )
        self.label_corner_selection_image.grid(row=10, column=3, sticky="s")

        self.images_tab_cornerselection = ctk.CTkFrame(self.images_tab)
        self.images_tab_cornerselection.grid(row=11, column=3, sticky="ns")

        self.images_tab_cornerselection.columnconfigure((1, 2), weight=1)
        self.images_tab_cornerselection.rowconfigure((1, 2), weight=1)

        self.selected_corner_image = ctk.StringVar()

        self.create_corner_radio_buttons(
            self.images_tab_cornerselection, self.selected_corner_image, "image"
        )

    def create_logo_image_offset_elements(self):
        """Create elements for the logo image offset."""
        self.label_offset_width_logo_image = ctk.CTkLabel(
            self.images_tab,
            text=f"Width Offset [Range: 0 to 100, Default: {OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_width_logo_image.grid(row=12, column=3, sticky="s")

        self.entry_offset_width_logo_image = ctk.CTkEntry(
            self.images_tab,
            placeholder_text="Width Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_width_logo_image.grid(row=13, column=3, sticky="n")

        self.label_offset_height_logo_image = ctk.CTkLabel(
            self.images_tab,
            text=f"Height Offset [Range: 0 to 100, Default: {OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_height_logo_image.grid(row=14, column=3, sticky="s")

        self.entry_offset_height_logo_image = ctk.CTkEntry(
            self.images_tab,
            placeholder_text="Height Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_height_logo_image.grid(row=15, column=3, sticky="n")

    def create_convert_format_elements(self):
        """Create elements for converting image format."""
        self.combobox_convert_images_format = ctk.CTkComboBox(
            self.images_tab,
            values=IMAGE_FORMATS,
            state="readonly",
            font=("Arial", 14),
            width=80,
        )
        self.combobox_convert_images_format.grid(row=16, column=3, sticky="ws")

        self.switch_convert_images_to_format = ctk.CTkSwitch(
            self.images_tab,
            text="Convert to format",
            font=("Arial", 14),
            text_color=DEFAULT_GRAY,
            command=self.switch_convert_images_to_format_actions,
        )
        self.switch_convert_images_to_format.grid(row=16, column=1, columnspan=3, sticky="s")

    def create_overwrite_checkbox(self, tab: ctk.CTkTabview, row: int):
        """Create the overwrite checkbox.

        Args:
            tab (ctk.CTkTabview): The tab to add the checkbox to.
            row (int): The row for the checkbox.

        """
        checkbox_clear_input_folder = ctk.CTkCheckBox(
            tab, text="Clear input folder", font=("Arial", 14)
        )
        checkbox_clear_input_folder.grid(row=row, column=1, columnspan=2, sticky="w", padx=50)

        checkbox_overwrite = ctk.CTkCheckBox(
            tab, text="Overwrite existing files in the output folder", font=("Arial", 14)
        )
        checkbox_overwrite.grid(row=row, column=1, columnspan=3)

        checkbox_clear_output_folder = ctk.CTkCheckBox(
            tab, text="Clear output folder", font=("Arial", 14)
        )
        checkbox_clear_output_folder.grid(row=row, column=3)

        match tab:
            case self.images_tab:
                self.checkbox_clear_input_folder_image = checkbox_clear_input_folder
                self.checkbox_overwrite_image = checkbox_overwrite
                self.checkbox_clear_output_folder_image = checkbox_clear_output_folder
            case self.videos_tab:
                self.checkbox_clear_input_folder_video = checkbox_clear_input_folder
                self.checkbox_overwrite_video = checkbox_overwrite
                self.checkbox_clear_output_folder_video = checkbox_clear_output_folder
            case _:
                raise ValueError(f"Tab {tab} is not supported.")

    def create_resize_button(self):
        """Create the resize button."""
        button = ctk.CTkButton(
            self.images_tab,
            text="Resize",
            font=("Arial", 16),
            command=self.check_values_and_paths_image_processor,
        )
        button.grid(row=20, column=1, columnspan=3, sticky="ns")

    def create_add_logo_button(self):
        """Create the add logo button."""
        button = ctk.CTkButton(
            self.videos_tab,
            text="Add Logo to Videos",
            font=("Arial", 16),
            command=self.check_values_and_paths_video_processor,
        )
        button.grid(row=17, column=1, columnspan=3, sticky="ns")

    def create_logo_video_elements(self):
        """Create elements for the logo video."""
        label_logo_title_video = ctk.CTkLabel(
            self.videos_tab, text="Logo parameters", font=("Arial", 20)
        )
        label_logo_title_video.grid(row=8, column=1, columnspan=3)

        self.create_logo_video_path_elements()
        self.create_logo_video_scale_elements()
        self.create_logo_video_corner_selection()
        self.create_logo_video_offset_elements()

    def create_logo_video_path_elements(self):
        """Create elements for the logo video path."""
        self.label_logo_video = ctk.CTkLabel(
            self.videos_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        self.label_logo_video.grid(row=9, column=2, sticky="s")

        self.entry_logo_video = ctk.CTkEntry(self.videos_tab, placeholder_text="Path to Logo Image")
        self.entry_logo_video.grid(row=10, column=2, sticky="ew")

        self.browse_button_logo_video = ctk.CTkButton(
            self.videos_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_video),
        )
        self.browse_button_logo_video.grid(row=11, column=2, sticky="n")

    def create_logo_video_scale_elements(self):
        """Create elements for the logo video scale."""
        self.label_scale_logo_video = ctk.CTkLabel(
            self.videos_tab,
            text=f"Scale of the logo image [Range: 1 to 100 , Default: {VIDEO_SCALE_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_scale_logo_video.grid(row=12, column=2, sticky="s")

        self.entry_scale_logo_video = ctk.CTkEntry(
            self.videos_tab,
            placeholder_text="Scale",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_scale_logo_video.grid(row=13, column=2, sticky="n")

    def create_logo_video_corner_selection(self):
        """Create elements for the logo video corner selection."""
        self.label_corner_selection_video = ctk.CTkLabel(
            self.videos_tab, text="Corner Selection", font=("Arial", 14)
        )
        self.label_corner_selection_video.grid(row=8, column=3, sticky="s")

        frame_corner_selection_video = ctk.CTkFrame(self.videos_tab)
        frame_corner_selection_video.grid(row=9, column=3, sticky="ns")

        frame_corner_selection_video.columnconfigure((1, 2), weight=1)
        frame_corner_selection_video.rowconfigure((1, 2), weight=1)

        self.selected_corner_video = ctk.StringVar()

        self.create_corner_radio_buttons(
            frame_corner_selection_video, self.selected_corner_video, "video"
        )

    def create_logo_video_offset_elements(self):
        """Create elements for the logo video offset."""
        self.label_offset_width_logo_video = ctk.CTkLabel(
            self.videos_tab,
            text=f"Width Offset [Range: -100 to 100, Default: {VIDEO_WIDTH_OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_width_logo_video.grid(row=10, column=3, sticky="s")

        self.entry_offset_width_logo_video = ctk.CTkEntry(
            self.videos_tab,
            placeholder_text="Width Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_width_logo_video.grid(row=11, column=3, sticky="n")

        self.label_offset_height_logo_video = ctk.CTkLabel(
            self.videos_tab,
            text=f"Height Offset [Range: -100 to 100, Default: {VIDEO_HEIGHT_OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_height_logo_video.grid(row=12, column=3, sticky="s")

        self.entry_offset_height_logo_video = ctk.CTkEntry(
            self.videos_tab,
            placeholder_text="Height Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_height_logo_video.grid(row=13, column=3, sticky="n")

    def create_corner_radio_buttons(
        self, frame: ctk.CTkFrame, variable: ctk.StringVar, prefix: str
    ):
        """Create radio buttons for corner selection.

        Args:
            frame (ctk.CTkFrame): The frame to add the radio buttons to.
            variable (ctk.StringVar): The variable to bind the radio buttons to.
            prefix (str): The prefix for the attribute names.

        """
        corners = [
            ("Top Left", 1, 1, f"radiobutton_topleft_{prefix}"),
            ("Top Right", 1, 2, f"radiobutton_topright_{prefix}"),
            ("Bottom Left", 2, 1, f"radiobutton_bottomleft_{prefix}"),
            ("Bottom Right", 2, 2, f"radiobutton_bottomright_{prefix}"),
        ]
        for text, row, col, attr_name in corners:
            radio_button = ctk.CTkRadioButton(frame, text=text, variable=variable, value=text)
            radio_button.grid(row=row, column=col, sticky="e" if "Right" in text else "w")
            setattr(self, attr_name, radio_button)

    def browse_path(self, entry: ctk.CTkEntry):
        """Browse and select a path for the given entry.

        Args:
            entry (ctk.CTkEntry): The entry widget to update with the selected path.

        """
        initial_dir = None
        current_path = entry.get()

        # Check if the current entry has a valid path and set the initial directory one level up
        if current_path:
            try:
                parent_path = Path(current_path).parent
                if parent_path.exists():
                    initial_dir = str(parent_path)
            except Exception as e:
                logging.error(f"Error processing path: {e}")

        try:
            if "Path to Logo Image" in entry.cget("placeholder_text"):
                file_string = filedialog.askopenfilename(initialdir=initial_dir)
                if file_string:  # Check if a file was selected
                    path = Path(file_string)
                else:
                    return
            else:
                directory = filedialog.askdirectory(initialdir=initial_dir)
                if directory:  # Check if a directory was selected
                    path = Path(directory)
                else:
                    return

            entry.delete(0, tk.END)
            entry.insert(tk.END, str(path))
        except IndexError:
            # Exception triggers if the user cancels the file dialog
            pass

    def toggle_logo_actions(self):
        """Toggle the state of logo-related UI elements based on the toggle state."""
        if self.toggle_logo_image.get() == 1:
            self.toggle_logo_image.configure(text_color=DEFAULT_WHITE)
            self.label_logo_image.configure(text_color=DEFAULT_WHITE)
            self.entry_logo_image.configure(state="normal", text_color=DEFAULT_WHITE)
            self.browse_button_logo_image.configure(state="normal")
            self.label_scale_logo_image.configure(text_color=DEFAULT_WHITE)
            self.entry_scale_logo_image.configure(state="normal", text_color=DEFAULT_WHITE)
            self.label_corner_selection_image.configure(text_color=DEFAULT_WHITE)
            self.radiobutton_bottomleft_image.configure(state="normal")
            self.radiobutton_bottomright_image.configure(state="normal")
            self.radiobutton_topleft_image.configure(state="normal")
            self.radiobutton_topright_image.configure(state="normal")
            self.label_offset_width_logo_image.configure(text_color=DEFAULT_WHITE)
            self.entry_offset_width_logo_image.configure(state="normal", text_color=DEFAULT_WHITE)
            self.label_offset_height_logo_image.configure(text_color=DEFAULT_WHITE)
            self.entry_offset_height_logo_image.configure(state="normal", text_color=DEFAULT_WHITE)
        else:
            self.toggle_logo_image.configure(text_color=DEFAULT_GRAY)
            self.label_logo_image.configure(text_color=DEFAULT_GRAY)
            self.entry_logo_image.configure(state="disabled", text_color=DEFAULT_GRAY)
            self.browse_button_logo_image.configure(state="disabled")
            self.label_scale_logo_image.configure(text_color=DEFAULT_GRAY)
            self.entry_scale_logo_image.configure(state="disabled", text_color=DEFAULT_GRAY)
            self.label_corner_selection_image.configure(text_color=DEFAULT_GRAY)
            self.radiobutton_bottomleft_image.configure(state="disabled")
            self.radiobutton_bottomright_image.configure(state="disabled")
            self.radiobutton_topleft_image.configure(state="disabled")
            self.radiobutton_topright_image.configure(state="disabled")
            self.label_offset_width_logo_image.configure(text_color=DEFAULT_GRAY)
            self.entry_offset_width_logo_image.configure(state="disabled", text_color=DEFAULT_GRAY)
            self.label_offset_height_logo_image.configure(text_color=DEFAULT_GRAY)
            self.entry_offset_height_logo_image.configure(state="disabled", text_color=DEFAULT_GRAY)

    def switch_convert_images_to_format_actions(self):
        """Toggle the state of image format conversion UI elements based on the switch state."""
        if self.switch_convert_images_to_format.get() == 1:
            self.combobox_convert_images_format.configure(state="readonly")
            self.switch_convert_images_to_format.configure(text_color=DEFAULT_WHITE)
        else:
            self.combobox_convert_images_format.configure(state="disabled")
            self.switch_convert_images_to_format.configure(text_color=DEFAULT_GRAY)

    def check_values_and_paths_image_processor(self):
        """Check the validity of input values and paths for the image processor."""
        input_folder_path = Path(self.entry_input_image.get())
        output_folder_path = Path(self.entry_output_image.get())
        logo_path = Path(self.entry_logo_image.get())

        if not input_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not output_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not logo_path.is_file():
            messagebox.showerror("Error", "Invalid path for logo image: No file recognized")
            return
        try:
            scale_value = int(self.entry_scale_logo_image.get())
            width_offset = int(self.entry_offset_width_logo_image.get())
            height_offset = int(self.entry_offset_height_logo_image.get())

            if not (1 <= scale_value <= 100):
                raise ValueError("Invalid scale value")
            if not (0 <= width_offset <= 100):
                raise ValueError("Invalid Width Offset value")
            if not (0 <= height_offset <= 100):
                raise ValueError("Invalid Height Offset value")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        # Check if the maximum pixel value is valid
        try:
            min_pixels = int(self.entry_num_pixels.get())
            if min_pixels <= 0:
                raise ValueError("Invalid pixel value")
        except ValueError:
            messagebox.showerror("Error", "Invalid pixel value")
            return

        overwrite_existing_images = self.checkbox_overwrite_image.get()
        add_logo = True if self.toggle_logo_image.get() == 1 else False
        selected_corner = self.selected_corner_image.get()
        convert_to_format = True if self.switch_convert_images_to_format.get() == 1 else False
        format = self.combobox_convert_images_format.get()
        clear_input_folder = True if self.checkbox_clear_input_folder_image.get() == 1 else False
        clear_output_folder = True if self.checkbox_clear_output_folder_image.get() == 1 else False

        ImageProcessor.resize_images(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            min_pixels=min_pixels,
            overwrite_existing_images=overwrite_existing_images,
            add_logo=add_logo,
            logo_path=logo_path,
            scale=scale_value,
            offset_logo_width=width_offset,
            offset_logo_height=height_offset,
            selected_corner=selected_corner,
            convert_to_format=convert_to_format,
            format=format,
            clear_input_folder=clear_input_folder,
            clear_output_folder=clear_output_folder,
        )

    def check_values_and_paths_video_processor(self):
        """Check the validity of input values and paths for the video processor."""
        input_folder_path = Path(self.entry_input_video.get())
        output_folder_path = Path(self.entry_output_video.get())
        logo_path = Path(self.entry_logo_video.get())

        if not input_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not output_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not logo_path.is_file():
            messagebox.showerror("Error", "Invalid path for logo image: No file recognized")
            return
        try:
            scale_value = int(self.entry_scale_logo_video.get())
            width_offset = int(self.entry_offset_width_logo_video.get())
            height_offset = int(self.entry_offset_height_logo_video.get())

            if not (1 <= scale_value <= 100):
                raise ValueError("Invalid scale value")
            if not (0 <= width_offset <= 100):
                raise ValueError("Invalid Width Offset value")
            if not (0 <= height_offset <= 100):
                raise ValueError("Invalid Height Offset value")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        overwrite_existing_videos = self.checkbox_overwrite_video.get()
        selected_corner = self.selected_corner_video.get()
        clear_input_folder = True if self.checkbox_clear_input_folder_video.get() == 1 else False
        clear_output_folder = True if self.checkbox_clear_output_folder_video.get() == 1 else False

        VideoProcessor.add_logo_to_video(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            overwrite_existing_videos=overwrite_existing_videos,
            logo_path=logo_path,
            scale=scale_value,
            offset_logo_width=width_offset,
            offset_logo_height=height_offset,
            selected_corner=selected_corner,
            clear_input_folder=clear_input_folder,
            clear_output_folder=clear_output_folder,
        )

    def insert_settings(self):
        """Insert settings into the UI elements."""
        settings = self.load_settings()
        images_settings = settings["image_processor"]
        videos_settings = settings["video_processor"]

        self._insert_image_settings(images_settings)
        self._insert_video_settings(videos_settings)

    def _insert_image_settings(self, images_settings: dict):
        """Insert image processor settings into the UI elements.

        Args:
            images_settings (dict): Dictionary containing image processor settings.

        """
        self.entry_input_image.insert(0, images_settings["input_folder_path"])
        self.entry_output_image.insert(0, images_settings["output_folder_path"])
        self.entry_num_pixels.insert(0, images_settings["number_pixels"])

        try:
            self.toggle_logo_image.select() if images_settings[
                "add_logo"
            ] else self.toggle_logo_image.deselect()
            self.entry_logo_image.insert(0, images_settings["logo_image_path"])
            self.entry_scale_logo_image.insert(0, images_settings["scale"])
            self.entry_offset_width_logo_image.insert(0, images_settings["width_offset"])
            self.entry_offset_height_logo_image.insert(0, images_settings["height_offset"])
            self.toggle_logo_actions()
        except Exception as e:
            logging.error(f"Error: {e}")

        match images_settings["logo_corner"]:
            case "Top Left":
                self.radiobutton_topleft_image.select()
            case "Top Right":
                self.radiobutton_topright_image.select()
            case "Bottom Left":
                self.radiobutton_bottomleft_image.select()
            case "Bottom Right":
                self.radiobutton_bottomright_image.select()

        try:
            self.switch_convert_images_to_format.select() if images_settings[
                "convert_to_format"
            ] else self.switch_convert_images_to_format.deselect()

            self.combobox_convert_images_format.set(images_settings["format"])
            self.switch_convert_images_to_format_actions()
        except Exception as e:
            logging.error(f"Error: {e}")
        try:
            self.checkbox_clear_input_folder_image.select() if images_settings[
                "clear_input_folder"
            ] else self.checkbox_clear_input_folder_image.deselect()
            self.checkbox_clear_output_folder_image.select() if images_settings[
                "clear_output_folder"
            ] else self.checkbox_clear_output_folder_image.deselect()
        except Exception as e:
            logging.error(f"Error: {e}")

    def _insert_video_settings(self, videos_settings: dict):
        """Insert video processor settings into the UI elements.

        Args:
            videos_settings (dict): Dictionary containing video processor settings.

        """
        self.entry_input_video.insert(0, videos_settings["input_folder_path"])
        self.entry_output_video.insert(0, videos_settings["output_folder_path"])
        self.entry_logo_video.insert(0, videos_settings["logo_image_path"])
        self.entry_scale_logo_video.insert(0, videos_settings["scale"])
        self.entry_offset_width_logo_video.insert(0, videos_settings["width_offset"])
        self.entry_offset_height_logo_video.insert(0, videos_settings["height_offset"])

        match videos_settings["logo_corner"]:
            case "Top Left":
                self.radiobutton_topleft_video.select()
            case "Top Right":
                self.radiobutton_topright_video.select()
            case "Bottom Left":
                self.radiobutton_bottomleft_video.select()
            case "Bottom Right":
                self.radiobutton_bottomright_video.select()

        try:
            self.checkbox_clear_input_folder_video.select() if videos_settings[
                "clear_input_folder"
            ] else self.checkbox_clear_input_folder_video.deselect()
            self.checkbox_clear_output_folder_video.select() if videos_settings[
                "clear_output_folder"
            ] else self.checkbox_clear_output_folder_video.deselect()
        except Exception as e:
            logging.error(f"Error: {e}")

    def create_progress_bar_image_processor(self, total_files: int):
        """Create a progress bar for the image processor.

        Args:
            total_files (int): Total number of files to process.

        """
        # Remove existing progress bar if it exists
        if hasattr(self, "progress_bar") and self.progress_bar.winfo_exists():
            self.progress_bar.destroy()

        if hasattr(self, "label_percent") and self.label_percent.winfo_exists():
            self.label_percent.destroy()

        self.percent = ctk.StringVar()

        self.label_percent = ctk.CTkLabel(
            master=self.images_tab, textvariable=self.percent, font=("Arial", 15)
        )
        self.label_percent.grid(row=22, column=1, columnspan=3, sticky="s")

        self.progress_bar = ttk.Progressbar(
            master=self.images_tab, orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.grid(row=23, column=1, columnspan=3, sticky="ew")

        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

    def update_progress_bar_image_processor(self, total_files: int, i: int):
        """Update the progress bar for the image processor.

        Args:
            total_files (int): Total number of files to process.
            i (int): Current file index.

        """
        self.progress_bar["value"] = i
        self.percent.set(f"{i}/{total_files} images processed ({int((i/total_files)*100)}%)")
        self.images_tab.update()

    def finish_progress_bar_image_processor(self, execution_time: float):
        """Finish the progress bar for the image processor.

        Args:
            execution_time (float): Total execution time in seconds.

        """
        self.percent.set(f"Done! Processed all images in {execution_time} seconds")
        self.images_tab.update()
        self.progress_bar.pack_forget()

    def create_progress_bar_video_processor(self, total_files: int):
        """Create a progress bar for the video processor.

        Args:
            total_files (int): Total number of files to process.

        """
        # Remove existing progress bar if it exists
        if hasattr(self, "progress_bar") and self.progress_bar.winfo_exists():
            self.progress_bar.destroy()

        if hasattr(self, "label_percent") and self.label_percent.winfo_exists():
            self.label_percent.destroy()

        self.percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(
            master=self.videos_tab, textvariable=self.percent, font=("Arial", 15)
        )
        label_percent.grid(row=20, column=1, columnspan=3, sticky="s")

        self.progress_bar = ttk.Progressbar(
            master=self.videos_tab, orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.grid(row=21, column=1, columnspan=3, sticky="ew")

        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

        self.update_progress_bar_video_processor(total_files, 0)

    def update_progress_bar_video_processor(self, total_files: int, i: int):
        """Update the progress bar for the video processor.

        Args:
            total_files (int): Total number of files to process.
            i (int): Current file index.

        """
        self.progress_bar["value"] = i
        self.percent.set(
            f"{i}/{total_files} videos processed ({int((i/total_files)*100)}%), see the command window for the estimated times..."
        )
        self.images_tab.update()

    def finish_progress_bar_video_processor(self, execution_time: float):
        """Finish the progress bar for the video processor.

        Args:
            execution_time (float): Total execution time in seconds.

        """
        self.percent.set(f"Done! Processed all videos in {execution_time} seconds")
        self.images_tab.update()
        self.progress_bar.pack_forget()

    @staticmethod
    def clear_folder(folder_path: str):
        """Clear the contents of a folder by removing and recreating it.

        Args:
            folder_path (str): Path to the folder to be cleared.

        """
        try:
            # Remove the entire folder
            shutil.rmtree(folder_path)
            # Recreate the folder
            os.makedirs(folder_path)
        except Exception as e:
            logging.info("Failed to clear folder %s. Reason: %s" % (folder_path, e))

    @staticmethod
    def validate_numeric_input(value_if_allowed: str) -> bool:
        """Validate if the provided input value is a valid numeric value or a specific string.

        This function is used to validate user input for numeric fields in the application, such as the number of pixels, scale, width offset, and height offset.

        Args:
            value_if_allowed (str): The input value to be validated.

        Returns:
            bool: True if the input value is valid, False otherwise.

        """
        if (
            value_if_allowed.isdigit()
            or value_if_allowed == ""
            or value_if_allowed == "Number of pixels"
            or value_if_allowed == "Scale"
            or value_if_allowed == "Width offset"
            or value_if_allowed == "Height offset"
        ):
            return True
        else:
            return False

    @staticmethod
    def load_settings() -> dict:
        """Load the settings from settings.json.

        Returns
            dict: Dictionary with all the settings from settings.json.

        """
        with open(SETTINGS_PATH, "r") as file:
            settings = json.load(file)
        return settings

    @staticmethod
    def save_settings(settings: dict):
        """Save the provided settings to the specified JSON settings file.

        Args:
            settings (dict): A dictionary containing the settings to be saved.

        """
        with open(SETTINGS_PATH, "w") as file:
            json.dump(settings, file, indent=4)

    @staticmethod
    def load_config() -> dict:
        """Load the config from config.json.

        Returns
            dict: Dictionary with all the config items from config.json.

        """
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
        return config


def create_config_json():
    """Create a default config.json file."""
    default_config_json_template = {
        "repo_url": "https://raw.githubusercontent.com/0DarkPhoenix/Add-Logo-Processor/main/",
        "version": "v1.4",
        "downgrade_version": "",
    }
    with open(CONFIG_PATH, "w") as file:
        json.dump(default_config_json_template, file, indent=4)


def create_settings_json():
    """Create a default settings.json file."""
    default_settings_json_template = {
        "image_processor": {
            "input_folder_path": "",
            "output_folder_path": "",
            "number_pixels": "",
            "add_logo": False,
            "logo_image_path": "",
            "scale": str(SCALE_DEFAULT),
            "width_offset": str(OFFSET_DEFAULT),
            "height_offset": str(OFFSET_DEFAULT),
            "logo_corner": LOGO_CORNER_DEFAULT,
            "convert_to_format": False,
            "format": IMAGE_FORMATS[0],
            "overwrite_existing_files": False,
            "clear_input_folder": False,
            "clear_output_folder": False,
        },
        "video_processor": {
            "input_folder_path": "",
            "output_folder_path": "",
            "logo_image_path": "",
            "scale": str(VIDEO_SCALE_DEFAULT),
            "width_offset": str(VIDEO_WIDTH_OFFSET_DEFAULT),
            "height_offset": str(VIDEO_HEIGHT_OFFSET_DEFAULT),
            "logo_corner": LOGO_CORNER_DEFAULT,
            "overwrite_existing_files": False,
            "clear_input_folder": False,
            "clear_output_folder": False,
        },
    }

    MainWindow.save_settings(default_settings_json_template)
    logging.info("Created settings.json")


async def check_version() -> None:
    """Check if there is a new version of the program available and asks if the user wants to update."""
    try:

        async def get_latest_version(repo_url):
            async with aiohttp.ClientSession() as session:
                async with session.get(repo_url + "version.txt") as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        return None

        config = MainWindow.load_config()

        repo_url = config["repo_url"]

        current_version = config["version"]
        latest_version = await get_latest_version(repo_url)

        if current_version is None or latest_version is None:
            logging.error("Failed to retrieve versions.")
            return

        logging.info(f"Current version: {current_version}")
        logging.info(f"Latest version: {latest_version}")

        if version.parse(latest_version) > version.parse(current_version):
            logging.info(f"New version available! ({current_version} -> {latest_version})")
            update_window = UpdateAvailableWindow()
            update_window.run(current_version, latest_version)
            update_window.mainloop()
        else:
            logging.info("You are on the most recent version!")

    except Exception as e:
        logging.error(f"Failed to check if a new version is available: {e}")
        logging.info(f"Please check the log file at {LOG_FILE_PATH} for more details.")


if __name__ == "__main__":
    logging.info(f"{'#'*10} Add Logo Processor application has started {'#'*10}")
    if not os.path.exists(CONFIG_PATH):
        create_config_json()

    if not os.path.exists(SETTINGS_PATH):
        create_settings_json()

    asyncio.run(check_version())

    main_window = MainWindow()
    main_window.run()
    main_window.mainloop()
