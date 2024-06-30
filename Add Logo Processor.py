import json
import logging
import os
import subprocess
import sys
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import cv2
import numpy as np
import requests
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip
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
def handle_exception(exc_type, exc_value, exc_traceback):
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

# ----------------------------------- v1.3 ----------------------------------- #
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

        label_update_question = ctk.CTkLabel(self, text="Would you like to update?", font=("Arial", 14))
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
    def save_image_as_webp(output_file, img, quality=80):
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
                position = int(new_width - logo_width - offset_logo_width), int(0 + offset_logo_height)
            case "Bottom Left":
                position = int(0 + offset_logo_width), int(new_height - logo_height - offset_logo_height)
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
        overwrite_output_images: bool,
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
            overwrite_output_images (bool): Whether to overwrite existing images.
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

        if not output_file.exists() or overwrite_output_images:
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
                    scale,
                    width,
                    height,
                    new_height,
                    new_width,
                    logo_img,
                    img,
                    selected_corner,
                    offset_logo_width,
                    offset_logo_height,
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
        overwrite_output_images: bool,
        add_logo: bool,
        logo_path: Path,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
        convert_to_format: bool,
        format: str,
    ):
        """Resizes multiple images and optionally adds a logo.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            min_pixels (int): The minimum number of pixels for the smallest side.
            overwrite_output_images (bool): Whether to overwrite existing images.
            add_logo (bool): Whether to add a logo to the images.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.
            convert_to_format (bool): Whether to convert the images to a different format.
            format (str): The format to which the images will be converted.

        """
        start_time = time.time()

        # Check if the min_pixels value hasn't changed since the last resize operation, otherwise it will overwrite all
        # images in the output folder
        try:
            settings = MainWindow.load_settings()
            settings_image_processor = settings["image_processor"]

            if settings_image_processor["number_pixels"] != min_pixels:
                overwrite_output_images = True

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
                }
            )
            MainWindow.save_settings(settings)
        except Exception as e:
            logging.error(f"Error: {e}")

        # Create a progress bar
        total_files = len(files)
        MainWindow.create_progress_bar_image_processor(main_window, total_files)

        # Resize images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    ImageProcessor.resize_image_operation,
                    input_folder_path,
                    output_folder_path,
                    file,
                    min_pixels,
                    overwrite_output_images,
                    add_logo,
                    logo_img,
                    scale,
                    offset_logo_width,
                    offset_logo_height,
                    selected_corner,
                    convert_to_format,
                    format,
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures), start=1):
                future.result()  # Wait for the resize_image operation to complete
                MainWindow.update_progress_bar_image_processor(main_window, total_files, i)

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
        overwrite_output_videos: bool,
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
            overwrite_output_videos (bool): Whether to overwrite existing videos.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.

        """
        output_file = output_folder_path / file.name

        if not output_file.exists() or overwrite_output_videos:
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
        overwrite_output_videos: bool,
        logo_path: Path,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        """Add a logo to multiple videos.

        Args:
            input_folder_path (Path): The path to the input folder.
            output_folder_path (Path): The path to the output folder.
            overwrite_output_videos (bool): Whether to overwrite existing videos.
            logo_path (Path): The path to the logo image.
            scale (int): The scale of the logo.
            offset_logo_width (int): The width offset for the logo.
            offset_logo_height (int): The height offset for the logo.
            selected_corner (str): The corner where the logo will be placed.

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

        # Display Progress Bar
        total_files = len(files)
        MainWindow.create_progress_bar_video_processor(main_window, total_files)

        # Add logo to video in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    VideoProcessor.add_logo_to_video_operation,
                    input_folder_path,
                    output_folder_path,
                    file,
                    overwrite_output_videos,
                    logo_path,
                    scale,
                    offset_logo_width,
                    offset_logo_height,
                    selected_corner,
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures), start=1):
                future.result()  # Wait for the add_logo_to_video_operation to complete
                MainWindow.update_progress_bar_video_processor(main_window, total_files, i)

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

        self.validate_numeric_input_cmd_command = (
            self.register(self.validate_numeric_input),
            "%P",
        )

        self.create_window_elements()
        self.insert_settings()

    def create_window_elements(self):
        """Create the elements for the main window."""
        main_title = ctk.CTkLabel(self, text="Image & Video Processor", font=("Arial", 28))
        main_title.place(relx=0.5, rely=0.03, anchor="center")

        # Tabs
        tabview = ctk.CTkTabview(self, fg_color="#242424")
        tabview.place(relx=0, rely=0.07, relwidth=1, relheight=0.93)

        self.images_tab = tabview.add("Images")
        self.videos_tab = tabview.add("Videos")

        # ---------------------------------------------------------------------------- #
        #                                Image Processor                               #
        # ---------------------------------------------------------------------------- #

        # Define grid
        self.images_tab.columnconfigure(1, weight=1)
        self.images_tab.columnconfigure(2, weight=1)
        self.images_tab.columnconfigure(3, weight=1)

        for i in range(24):
            self.images_tab.rowconfigure(i, weight=1)

        # Input folder
        label_input = ctk.CTkLabel(self.images_tab, text="Input Folder", font=("Arial", 14))
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_image = ctk.CTkEntry(self.images_tab, placeholder_text="Path of input folder", width=850)
        self.entry_input_image.grid(row=2, column=1, columnspan=3)

        self.browse_button_input = ctk.CTkButton(
            self.images_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_image),
        )
        self.browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(self.images_tab, text="Output Folder", font=("Arial", 14))
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_image = ctk.CTkEntry(self.images_tab, placeholder_text="Path of output folder", width=850)
        self.entry_output_image.grid(row=5, column=1, columnspan=3)

        browse_button_output = ctk.CTkButton(
            self.images_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(self.entry_output_image),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Number of pixels
        label_num_pixels = ctk.CTkLabel(
            self.images_tab,
            text="Maximum number of pixels of the smallest side",
            font=("Arial", 14),
        )
        label_num_pixels.grid(row=7, column=1, columnspan=3, sticky="s")

        self.entry_num_pixels = ctk.CTkEntry(
            self.images_tab,
            placeholder_text="Number of pixels",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_num_pixels.grid(row=8, column=1, columnspan=3)

        # Logo image
        self.frame_logo_image = ctk.CTkFrame(
            self.images_tab,
            fg_color="transparent",
        )
        self.frame_logo_image.grid(row=10, rowspan=6, column=1, columnspan=3, sticky="nsew")

        self.toggle_logo_image = ctk.CTkSwitch(
            self.images_tab,
            text="Add Logo to Image",
            font=("Arial", 18),
            command=self.toggle_logo_actions,
        )
        self.toggle_logo_image.grid(row=10, column=1, columnspan=3)

        self.label_logo_image = ctk.CTkLabel(self.images_tab, text="Path to Logo Image", font=("Arial", 14))
        self.label_logo_image.grid(row=11, column=2, sticky="s")

        self.entry_logo_image = ctk.CTkEntry(self.images_tab, placeholder_text="Path to Logo Image")
        self.entry_logo_image.grid(row=12, column=2, sticky="ew")

        self.browse_button_logo_image = ctk.CTkButton(
            self.images_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_image),
        )
        self.browse_button_logo_image.grid(row=13, column=2, sticky="n")

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

        # Logo corner selection
        self.label_corner_selection_image = ctk.CTkLabel(self.images_tab, text="Corner Selection", font=("Arial", 14))
        self.label_corner_selection_image.grid(row=10, column=3, sticky="s")

        self.images_tab_cornerselection = ctk.CTkFrame(self.images_tab)
        self.images_tab_cornerselection.grid(row=11, column=3, sticky="ns")

        self.images_tab_cornerselection.columnconfigure((1, 2), weight=1)
        self.images_tab_cornerselection.rowconfigure((1, 2), weight=1)

        self.selected_corner_image = ctk.StringVar()

        self.radiobutton_topleft_image = ctk.CTkRadioButton(
            self.images_tab_cornerselection,
            text="Top Left",
            variable=self.selected_corner_image,
            value="Top Left",
        )
        self.radiobutton_topleft_image.grid(row=1, column=1)

        self.radiobutton_topright_image = ctk.CTkRadioButton(
            self.images_tab_cornerselection,
            text="Top Right",
            variable=self.selected_corner_image,
            value="Top Right",
        )
        self.radiobutton_topright_image.grid(row=1, column=2, sticky="e")

        self.radiobutton_bottomleft_image = ctk.CTkRadioButton(
            self.images_tab_cornerselection,
            text="Bottom Left",
            variable=self.selected_corner_image,
            value="Bottom Left",
        )
        self.radiobutton_bottomleft_image.grid(row=2, column=1)

        self.radiobutton_bottomright_image = ctk.CTkRadioButton(
            self.images_tab_cornerselection,
            text="Bottom Right",
            variable=self.selected_corner_image,
            value="Bottom Right",
        )
        self.radiobutton_bottomright_image.grid(row=2, column=2, sticky="e")

        # Offset logo image
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

        # Convert images to png
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

        # Overwrite output images
        self.checkbox_overwrite_image = ctk.CTkCheckBox(
            self.images_tab,
            text="Overwrite existing images in the output folder",
            font=("Arial", 14),
        )
        self.checkbox_overwrite_image.grid(row=18, column=1, columnspan=3)

        # Resize button
        button_start_image_processor = ctk.CTkButton(
            self.images_tab,
            text="Resize",
            font=("Arial", 16),
            command=self.check_values_and_paths_image_processor,
        )
        button_start_image_processor.grid(row=20, column=1, columnspan=3, sticky="ns")

        # ---------------------------------------------------------------------------- #
        #                                Video Processor                               #
        # ---------------------------------------------------------------------------- #

        # Define grid
        self.videos_tab.columnconfigure(1, weight=1)
        self.videos_tab.columnconfigure(2, weight=1)
        self.videos_tab.columnconfigure(3, weight=1)

        for i in range(22):
            self.videos_tab.rowconfigure(i, weight=1)

        # Input folder
        label_input = ctk.CTkLabel(master=self.videos_tab, text="Input Folder", font=("Arial", 14))
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Path of input folder", width=850
        )
        self.entry_input_video.grid(row=2, column=1, columnspan=3)

        self.browse_button_input = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_video),
        )
        self.browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(master=self.videos_tab, text="Output Folder", font=("Arial", 14))
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Path of output folder", width=850
        )
        self.entry_output_video.grid(row=5, column=1, columnspan=3)

        browse_button_output = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(self.entry_output_video),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Logo image
        label_logo_title_video = ctk.CTkLabel(master=self.videos_tab, text="Logo parameters", font=("Arial", 20))
        label_logo_title_video.grid(row=8, column=1, columnspan=3)

        self.label_logo_video = ctk.CTkLabel(master=self.videos_tab, text="Path to Logo Image", font=("Arial", 14))
        self.label_logo_video.grid(row=9, column=2, sticky="s")

        self.entry_logo_video = ctk.CTkEntry(master=self.videos_tab, placeholder_text="Path to Logo Image")
        self.entry_logo_video.grid(row=10, column=2, sticky="ew")

        self.browse_button_logo_video = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_video),
        )
        self.browse_button_logo_video.grid(row=11, column=2, sticky="n")

        self.label_scale_logo_video = ctk.CTkLabel(
            master=self.videos_tab,
            text=f"Scale of the logo image [Range: 1 to 100 , Default: {VIDEO_SCALE_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_scale_logo_video.grid(row=12, column=2, sticky="s")

        self.entry_scale_logo_video = ctk.CTkEntry(
            master=self.videos_tab,
            placeholder_text="Scale",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_scale_logo_video.grid(row=13, column=2, sticky="n")

        # Logo corner selection
        self.label_corner_selection_video = ctk.CTkLabel(
            master=self.videos_tab, text="Corner Selection", font=("Arial", 14)
        )
        self.label_corner_selection_video.grid(row=8, column=3, sticky="s")

        frame_corner_selection_video = ctk.CTkFrame(master=self.videos_tab)
        frame_corner_selection_video.grid(row=9, column=3, sticky="ns")

        frame_corner_selection_video.columnconfigure((1, 2), weight=1)
        frame_corner_selection_video.rowconfigure((1, 2), weight=1)

        self.selected_corner_video = ctk.StringVar()

        self.radiobutton_topleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection_video,
            text="Top Left",
            variable=self.selected_corner_video,
            value="Top Left",
        )
        self.radiobutton_topleft_video.grid(row=1, column=1)

        self.radiobutton_topright_video = ctk.CTkRadioButton(
            master=frame_corner_selection_video,
            text="Top Right",
            variable=self.selected_corner_video,
            value="Top Right",
        )
        self.radiobutton_topright_video.grid(row=1, column=2, sticky="e")

        self.radiobutton_bottomleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection_video,
            text="Bottom Left",
            variable=self.selected_corner_video,
            value="Bottom Left",
        )
        self.radiobutton_bottomleft_video.grid(row=2, column=1)

        self.radiobutton_bottomright_video = ctk.CTkRadioButton(
            master=frame_corner_selection_video,
            text="Bottom Right",
            variable=self.selected_corner_video,
            value="Bottom Right",
        )
        self.radiobutton_bottomright_video.grid(row=2, column=2, sticky="e")

        # Offset logo image
        self.label_offset_width_logo_video = ctk.CTkLabel(
            master=self.videos_tab,
            text=f"Width Offset [Range: -100 to 100, Default: {VIDEO_WIDTH_OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_width_logo_video.grid(row=10, column=3, sticky="s")

        self.entry_offset_width_logo_video = ctk.CTkEntry(
            master=self.videos_tab,
            placeholder_text="Width Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_width_logo_video.grid(row=11, column=3, sticky="n")

        self.label_offset_height_logo_video = ctk.CTkLabel(
            master=self.videos_tab,
            text=f"Height Offset [Range: -100 to 100, Default: {VIDEO_HEIGHT_OFFSET_DEFAULT}]",
            font=("Arial", 14),
        )
        self.label_offset_height_logo_video.grid(row=12, column=3, sticky="s")

        self.entry_offset_height_logo_video = ctk.CTkEntry(
            master=self.videos_tab,
            placeholder_text="Height Offset",
            validate="key",
            validatecommand=self.validate_numeric_input_cmd_command,
        )
        self.entry_offset_height_logo_video.grid(row=13, column=3, sticky="n")

        # Overwrite output videos
        self.checkbox_overwrite_video = ctk.CTkCheckBox(
            master=self.videos_tab,
            text="Overwrite existing videos in the output folder",
            font=("Arial", 14),
        )
        self.checkbox_overwrite_video.grid(row=15, column=1, columnspan=3)

        # Add Logo to Videos button
        button_start_video_processor = ctk.CTkButton(
            master=self.videos_tab,
            text="Add Logo to Videos",
            font=("Arial", 16),
            command=self.check_values_and_paths_video_processor,
        )
        button_start_video_processor.grid(row=17, column=1, columnspan=3, sticky="ns")

    def browse_path(self, entry):
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
        logo_image = Path(self.entry_logo_image.get())

        if not input_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not output_folder_path.exists():
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not logo_image.is_file():
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

        overwrite_output_images = self.checkbox_overwrite_image.get()
        add_logo = True if self.toggle_logo_image.get() == 1 else False
        selected_corner = self.selected_corner_image.get()
        convert_to_format = True if self.switch_convert_images_to_format.get() == 1 else False
        format = self.combobox_convert_images_format.get()

        ImageProcessor.resize_images(
            input_folder_path,
            output_folder_path,
            min_pixels,
            overwrite_output_images,
            add_logo,
            logo_image,
            scale_value,
            width_offset,
            height_offset,
            selected_corner,
            convert_to_format,
            format,
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

        overwrite_output_images = self.checkbox_overwrite_video.get()
        selected_corner = self.selected_corner_video.get()

        VideoProcessor.add_logo_to_video(
            input_folder_path,
            output_folder_path,
            overwrite_output_images,
            logo_path,
            scale_value,
            width_offset,
            height_offset,
            selected_corner,
        )

    def insert_settings(self):
        """Insert settings into the UI elements."""
        settings = self.load_settings()
        images_settings = settings["image_processor"]
        videos_settings = settings["video_processor"]

        self._insert_image_settings(images_settings)
        self._insert_video_settings(videos_settings)

    def _insert_image_settings(self, images_settings):
        """Insert image processor settings into the UI elements.

        Args:
            images_settings (dict): Dictionary containing image processor settings.

        """
        self.entry_input_image.insert(0, images_settings["input_folder_path"])
        self.entry_output_image.insert(0, images_settings["output_folder_path"])
        self.entry_num_pixels.insert(0, images_settings["number_pixels"])

        try:
            self.toggle_logo_image.select() if images_settings["add_logo"] else self.toggle_logo_image.deselect()
            self.entry_logo_image.insert(0, images_settings["logo_image_path"])
            self.entry_scale_logo_image.insert(0, images_settings["scale"])
            self.entry_offset_width_logo_image.insert(0, images_settings["width_offset"])
            self.entry_offset_height_logo_image.insert(0, images_settings["height_offset"])
            self.toggle_logo_actions()
        except Exception as e:
            print(f"Error: {e}")

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
            print(images_settings["format"])
            self.combobox_convert_images_format.set(images_settings["format"])
            self.switch_convert_images_to_format_actions()
        except Exception as e:
            print(f"Error: {e}")

    def _insert_video_settings(self, videos_settings):
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

    def create_progress_bar_image_processor(self, total_files: int):
        """Create a progress bar for the image processor.

        Args:
            total_files (int): Total number of files to process.

        """
        self.percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(master=self.images_tab, textvariable=self.percent, font=("Arial", 15))
        label_percent.grid(row=22, column=1, columnspan=3, sticky="s")

        self.progress_bar = ttk.Progressbar(master=self.images_tab, orient="horizontal", length=400, mode="determinate")
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
        self.percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(master=self.videos_tab, textvariable=self.percent, font=("Arial", 15))
        label_percent.grid(row=20, column=1, columnspan=3, sticky="s")

        self.progress_bar = ttk.Progressbar(master=self.videos_tab, orient="horizontal", length=400, mode="determinate")
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
        "version": "v1.3",
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
        },
        "video_processor": {
            "input_folder_path": "",
            "output_folder_path": "",
            "logo_image_path": "",
            "scale": str(VIDEO_SCALE_DEFAULT),
            "width_offset": str(VIDEO_WIDTH_OFFSET_DEFAULT),
            "height_offset": str(VIDEO_HEIGHT_OFFSET_DEFAULT),
            "logo_corner": LOGO_CORNER_DEFAULT,
        },
    }

    MainWindow.save_settings(default_settings_json_template)
    logging.info("Created settings.json")


def check_version() -> None:
    """Check if there is a new version of the program available and asks if the user wants to update."""
    try:

        def get_latest_version(repo_url):
            response = requests.get(repo_url + "version.txt")
            if response.status_code == 200:
                return response.text.strip()
            else:
                return None

        config = MainWindow.load_config()

        repo_url = config["repo_url"]

        current_version = config["version"]
        latest_version = get_latest_version(repo_url)

        if current_version is None or latest_version is None:
            logging.error("Failed to retrieve versions.")
            return

        logging.info(f"Current version: {current_version}")
        logging.info(f"Latest version: {latest_version}")

        if latest_version > current_version:
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

    check_version()

    main_window = MainWindow()
    main_window.run()
    main_window.mainloop()
