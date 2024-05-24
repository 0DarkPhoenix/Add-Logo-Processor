import json
import os
import sys
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import cv2
import numpy as np
from moviepy.editor import (  # Installed Pillow version 9.5, otherwise it will give an Antialiasing error when using the most current version
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
)

global main_window  # main_window gets defined if __name__ == "__main__"

if getattr(sys, "frozen", False):
    # If the application is run as a bundled executable, use the directory of the executable
    MAIN_PATH = os.path.dirname(sys.executable)
else:
    # Otherwise, just use the normal directory where the script resides
    MAIN_PATH = os.path.abspath(os.path.dirname(__file__))

SETTINGS_PATH = Path(MAIN_PATH, "settings.json")

# TODO: Make duplicate elements in gui dynamic by giving them the ability to be displayed in both tabs while keeping their different functions in their individual tabs (?v1.x)

# TODO: Add a togglable function which can convert images to png format (v1.0)
# TODO: Write code for an update checker (v1.0)


class ImageProcessor:

    @staticmethod
    def add_logo_operation(
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
    ):
        logo_height, logo_width, _ = logo_img.shape
        logo_aspectratio = logo_width / logo_height
        scale_factor = scale / 100

        if width < height:
            logo_height = new_height * scale_factor
            logo_width = (
                logo_height * logo_aspectratio if logo_aspectratio != 1 else logo_height
            )
        else:
            logo_width = new_width * scale_factor
            logo_height = (
                logo_width / logo_aspectratio if logo_aspectratio != 1 else logo_width
            )

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
                position = int(new_width - logo_width - offset_logo_width), int(
                    0 + offset_logo_height
                )
            case "Bottom Left":
                position = int(0 + offset_logo_width), int(
                    new_height - logo_height - offset_logo_height
                )
            case "Bottom Right":
                position = int(new_width - logo_width - offset_logo_width), int(
                    new_height - logo_height - offset_logo_height
                )

        x_offset, y_offset = position
        y1, y2 = y_offset, y_offset + logo_img.shape[0]
        x1, x2 = x_offset, x_offset + logo_img.shape[1]

        alpha_s = logo_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (
                alpha_s * logo_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c]
            )

    @staticmethod
    def resize_image_operation(
        input_folder_path: str,
        output_folder_path: str,
        file: str,
        min_pixels: int,
        overwrite_output_images: bool,
        add_logo: bool,
        logo_img: np.ndarray,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        output_file = Path(output_folder_path, file)

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

            cv2.imwrite(str(output_file), img)

    @staticmethod
    def resize_images(
        input_folder_path: str,
        output_folder_path: str,
        min_pixels: int,
        overwrite_output_images: bool,
        add_logo: bool,
        logo_path: str,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        start_time = time.time()

        # Check if the min_pixels value hasn't changed since the last resize operation, otherwise it will overwrite all images in the output folder
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

        logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

        # Save all values to settings.json
        settings_image_processor.update(
            {
                "input_folder_path": input_folder_path,
                "output_folder_path": output_folder_path,
                "number_pixels": min_pixels,
                "add_logo": add_logo,
                "logo_image_path": logo_path,
                "scale": scale,
                "width_offset": offset_logo_width,
                "height_offset": offset_logo_height,
                "logo_corner": selected_corner,
            }
        )
        MainWindow.save_settings(settings)

        # Create a progress bar
        total_files = len(files)
        MainWindow.create_progress_bar_image_processor(main_window, total_files)

        # Resize images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, file in enumerate(files, start=1):
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
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures), start=1):
                future.result()  # Wait for the resize_image operation to complete
                MainWindow.update_progress_bar_image_processor(
                    main_window, total_files, i
                )

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        print(f"All images were processed in {execution_time} seconds")
        MainWindow.finish_progress_bar_image_processor(main_window, execution_time)


class VideoProcessor:

    @staticmethod
    def add_logo_to_video_operation(
        input_folder_path: str,
        output_folder_path: str,
        file,
        overwrite_output_videos: bool,
        logo_path: str,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        output_file = Path(output_folder_path) / file.name

        if not output_file.exists() or overwrite_output_videos:
            # Load video clip
            video_path = Path(input_folder_path) / file.name
            video_clip = VideoFileClip(str(video_path))

            # Load the logo image and resize it to fit the video dimensions
            logo_clip = ImageClip(logo_path)
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
            video_clip = CompositeVideoClip(
                [video_clip, logo_clip.set_position(position)]
            )

            # Set the output path and save the modified video clip
            video_clip.write_videofile(
                str(output_file), codec="libx264"
            )  # Convert Path to string

    def add_logo_to_video(
        input_folder_path: str,
        output_folder_path: str,
        overwrite_output_videos: bool,
        logo_path: str,
        scale: int,
        offset_logo_width: int,
        offset_logo_height: int,
        selected_corner: str,
    ):
        start_time = time.time()

        # Set the input and output folders

        files = sorted(
            os.scandir(input_folder_path), key=lambda f: f.stat().st_size, reverse=True
        )

        settings = MainWindow.load_settings()
        settings_video_processor = settings["video_processor"]
        settings_video_processor.update(
            {
                "input_folder_path": input_folder_path,
                "output_folder_path": output_folder_path,
                "logo_image_path": logo_path,
                "scale": scale,
                "width_offset": offset_logo_width,
                "height_offset": offset_logo_height,
                "logo_corner": selected_corner,
            }
        )
        MainWindow.save_settings(settings)

        # Display Progress Bar
        total_files = len(files)
        MainWindow.create_progress_bar_video_processor(main_window, total_files)

        # Add logo to video in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, file in enumerate(files, start=1):
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
                future.result()  # Wait for the resize_image operation to complete
                MainWindow.update_progress_bar_video_processor(
                    main_window, total_files, i
                )

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        print(f"Processed all videos in {execution_time} seconds")
        MainWindow.finish_progress_bar_video_processor(main_window, execution_time)


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

    def run(self):
        self.title("Image & Video Processor")
        self.geometry("1000x1000")

        self.create_window_elements()
        self.insert_settings()

    def create_window_elements(self):
        main_title = ctk.CTkLabel(
            self, text="Image & Video Processor", font=("Arial", 28)
        )
        main_title.place(relx=0.35, rely=0.03)

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
            self.images_tab.rowconfigure(i, weight=1)  # Create row 1 to 20

        # Input folder
        label_input = ctk.CTkLabel(
            self.images_tab, text="Input Folder", font=("Arial", 14)
        )
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Path of input folder"
        )
        self.entry_input_image.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            self.images_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_image),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(
            self.images_tab, text="Output Folder", font=("Arial", 14)
        )
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Path of output folder"
        )
        self.entry_output_image.grid(row=5, column=1, columnspan=3, sticky="ew")

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
            self.images_tab, placeholder_text="Number of pixels"
        )
        self.entry_num_pixels.grid(row=8, column=1, columnspan=3)

        # Logo image
        self.frame_logo = ctk.CTkFrame(
            self.images_tab,
            border_width=3,
            border_color="dark blue",
            fg_color="transparent",
        )
        self.frame_logo.grid(row=10, rowspan=6, column=1, columnspan=3, sticky="nsew")

        self.toggle_logo = ctk.CTkSwitch(
            self.images_tab,
            text="Add Logo to Image",
            font=("Arial", 18),
            command=self.update_frame_logo_border_color,
        )
        self.toggle_logo.grid(row=10, column=1, columnspan=3)

        label_logo = ctk.CTkLabel(
            self.images_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        label_logo.grid(row=11, column=2, sticky="s")

        self.entry_logo_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Path to Logo Image"
        )
        self.entry_logo_image.grid(
            row=12, column=2, sticky="ew"
        )  # TODO: Add functionality to see a popup with the entire path if it exceeds a specified amount of characters, which makes it easier to read really long path names (v1.1)

        browse_button_input = ctk.CTkButton(
            self.images_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_image),
        )
        browse_button_input.grid(row=13, column=2, sticky="n")

        label_scale_logo = ctk.CTkLabel(
            self.images_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 10]",
            font=("Arial", 14),
        )
        label_scale_logo.grid(row=14, column=2, sticky="s")

        self.entry_scale_logo_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Scale"
        )
        self.entry_scale_logo_image.grid(row=15, column=2, sticky="n")

        # Logo corner selection
        label_corner_selection = ctk.CTkLabel(
            self.images_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_corner_selection.grid(row=10, column=3, sticky="s")

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
        label_offset_width_logo = ctk.CTkLabel(
            self.images_tab,
            text="Width Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_width_logo.grid(row=12, column=3, sticky="s")

        self.entry_offset_width_logo_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Width Offset"
        )
        self.entry_offset_width_logo_image.grid(row=13, column=3, sticky="n")

        label_offset_height_logo = ctk.CTkLabel(
            self.images_tab,
            text="Height Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_height_logo.grid(row=14, column=3, sticky="s")

        self.entry_offset_height_logo_image = ctk.CTkEntry(
            self.images_tab, placeholder_text="Height Offset"
        )
        self.entry_offset_height_logo_image.grid(row=15, column=3, sticky="n")

        # Overwrite output images
        self.checkbox_overwrite_image = ctk.CTkCheckBox(
            self.images_tab,
            text="Overwrite existing images in the output folder",
            font=("Arial", 14),
        )
        self.checkbox_overwrite_image.grid(row=17, column=1, columnspan=3)

        # Resize button
        button = ctk.CTkButton(
            self.images_tab,
            text="Resize",
            font=("Arial", 16),
            command=self.check_values_and_paths_image_processor,
        )
        button.grid(row=19, column=1, columnspan=3, sticky="ns")

        # ---------------------------------------------------------------------------- #
        #                                Video Processor                               #
        # ---------------------------------------------------------------------------- #

        # Define grid
        self.videos_tab.columnconfigure(1, weight=1)
        self.videos_tab.columnconfigure(2, weight=1)
        self.videos_tab.columnconfigure(3, weight=1)

        for i in range(20):
            self.videos_tab.rowconfigure(i, weight=1)

        # Input folder
        label_input = ctk.CTkLabel(
            master=self.videos_tab, text="Input Folder", font=("Arial", 14)
        )
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Path of input folder"
        )
        self.entry_input_video.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_video),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(
            master=self.videos_tab, text="Output Folder", font=("Arial", 14)
        )
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Path of output folder"
        )
        self.entry_output_video.grid(row=5, column=1, columnspan=3, sticky="ew")

        browse_button_output = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(self.entry_output_video),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Logo image
        label_logo_title = ctk.CTkLabel(
            master=self.videos_tab, text="Logo parameters", font=("Arial", 20)
        )
        label_logo_title.grid(row=8, column=1, columnspan=3)

        label_logo = ctk.CTkLabel(
            master=self.videos_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        label_logo.grid(row=9, column=2, sticky="s")

        self.entry_logo_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Path to Logo Image"
        )
        self.entry_logo_video.grid(row=10, column=2, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=self.videos_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_video),
        )
        browse_button_input.grid(row=11, column=2, sticky="n")

        label_scale_logo = ctk.CTkLabel(
            master=self.videos_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 13]",
            font=("Arial", 14),
        )
        label_scale_logo.grid(row=12, column=2, sticky="s")

        self.entry_scale_logo_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Scale"
        )
        self.entry_scale_logo_video.grid(row=13, column=2, sticky="n")

        # Logo corner selection
        label_corner_selection = ctk.CTkLabel(
            master=self.videos_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_corner_selection.grid(row=8, column=3, sticky="s")

        frame_corner_selection = ctk.CTkFrame(master=self.videos_tab)
        frame_corner_selection.grid(row=9, column=3, sticky="ns")

        frame_corner_selection.columnconfigure((1, 2), weight=1)
        frame_corner_selection.rowconfigure((1, 2), weight=1)

        self.selected_corner_video = ctk.StringVar()

        self.radiobutton_topleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Top Left",
            variable=self.selected_corner_video,
            value="Top Left",
        )
        self.radiobutton_topleft_video.grid(row=1, column=1)

        self.radiobutton_topright_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Top Right",
            variable=self.selected_corner_video,
            value="Top Right",
        )
        self.radiobutton_topright_video.grid(row=1, column=2, sticky="e")

        self.radiobutton_bottomleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Bottom Left",
            variable=self.selected_corner_video,
            value="Bottom Left",
        )
        self.radiobutton_bottomleft_video.grid(row=2, column=1)

        self.radiobutton_bottomright_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Bottom Right",
            variable=self.selected_corner_video,
            value="Bottom Right",
        )
        self.radiobutton_bottomright_video.grid(row=2, column=2, sticky="e")

        # Offset logo image
        label_offset_width_logo = ctk.CTkLabel(
            master=self.videos_tab,
            text="Width Offset [Range: -100 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_width_logo.grid(row=10, column=3, sticky="s")

        self.entry_offset_width_logo_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Width Offset"
        )
        self.entry_offset_width_logo_video.grid(row=11, column=3, sticky="n")

        label_offset_height_logo = ctk.CTkLabel(
            master=self.videos_tab,
            text="Height Offset [Range: -100 to 100, Default: 15]",
            font=("Arial", 14),
        )
        label_offset_height_logo.grid(row=12, column=3, sticky="s")

        self.entry_offset_height_logo_video = ctk.CTkEntry(
            master=self.videos_tab, placeholder_text="Height Offset"
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
        button = ctk.CTkButton(
            master=self.videos_tab,
            text="Add Logo to Videos",
            font=("Arial", 16),
            command=self.check_values_and_paths_video_processor,
        )
        button.grid(row=17, column=1, columnspan=3, sticky="ns")

    def browse_path(self, entry):
        try:
            if "Path to Logo Image" in entry.cget("placeholder_text"):
                file_string = filedialog.askopenfile()
                path = os.path.abspath(str(file_string).split("'")[1])
            else:
                path = filedialog.askdirectory()
            entry.delete(0, tk.END)
            entry.insert(tk.END, path)
        except IndexError:
            # Exception triggers if the user cancels the file dialog
            pass

    def check_values_and_paths_image_processor(self):
        input_folder_path = self.entry_input_image.get()
        output_folder_path = self.entry_output_image.get()
        logo_image = self.entry_logo_image.get()

        if not os.path.isdir(input_folder_path):
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not os.path.isdir(output_folder_path):
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not os.path.isfile(logo_image):
            messagebox.showerror(
                "Error", "Invalid path for logo image: No file recognized"
            )
            return
        scale_value = int(self.entry_scale_logo_image.get())
        width_offset = int(self.entry_offset_width_logo_image.get())
        height_offset = int(self.entry_offset_height_logo_image.get())

        if not (1 <= scale_value <= 100):
            messagebox.showerror("Error", "Invalid scale value")
            return
        if not (0 <= width_offset <= 100):
            messagebox.showerror("Error", "Invalid Width Offset value")
            return
        if not (0 <= height_offset <= 100):
            messagebox.showerror("Error", "Invalid Height Offset value")
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
        add_logo = True if self.toggle_logo.get() == 1 else False
        selected_corner = self.selected_corner_image.get()

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
        )

    def check_values_and_paths_video_processor(self):
        input_folder_path = self.entry_input_video.get()
        output_folder_path = self.entry_output_video.get()
        logo_path = self.entry_logo_video.get()

        if not os.path.isdir(input_folder_path):
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not os.path.isdir(output_folder_path):
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not os.path.isfile(logo_path):
            messagebox.showerror(
                "Error", "Invalid path for logo image: No file recognized"
            )
            return
        scale_value = int(self.entry_scale_logo_video.get())
        width_offset = int(self.entry_offset_width_logo_video.get())
        height_offset = int(self.entry_offset_height_logo_video.get())

        if not (1 <= scale_value <= 100):
            messagebox.showerror("Error", "Invalid scale value")
            return
        if not (0 <= width_offset <= 100):
            messagebox.showerror("Error", "Invalid Width Offset value")
            return
        if not (0 <= height_offset <= 100):
            messagebox.showerror("Error", "Invalid Height Offset value")
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

    def update_frame_logo_border_color(self):
        if self.toggle_logo.get() == 1:
            self.frame_logo.configure(border_color="darkblue")
        else:
            self.frame_logo.configure(border_color="black")

    def insert_settings(self):
        settings = self.load_settings()
        images_settings = settings["image_processor"]
        videos_settings = settings["video_processor"]

        # Image Processor
        self.entry_input_image.insert(0, images_settings["input_folder_path"])
        self.entry_output_image.insert(0, images_settings["output_folder_path"])
        self.entry_num_pixels.insert(0, images_settings["number_pixels"])
        try:
            if images_settings["add_logo"]:
                self.toggle_logo.select()
            else:
                self.toggle_logo.deselect()
        except:
            pass
        self.entry_logo_image.insert(0, images_settings["logo_image_path"])
        self.entry_scale_logo_image.insert(0, images_settings["scale"])
        self.entry_offset_width_logo_image.insert(0, images_settings["width_offset"])
        self.entry_offset_height_logo_image.insert(0, images_settings["height_offset"])
        match images_settings["logo_corner"]:
            case "Top Left":
                self.radiobutton_topleft_image.select()
            case "Top Right":
                self.radiobutton_topright_image.select()
            case "Bottom Left":
                self.radiobutton_bottomleft_image.select()
            case "Bottom Right":
                self.radiobutton_bottomright_image.select()

        # Video Processor
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
        self.percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(
            master=self.images_tab, textvariable=self.percent, font=("Arial", 15)
        )
        label_percent.grid(row=22, column=1, columnspan=3, sticky="s")

        self.progress_bar = ttk.Progressbar(
            master=self.images_tab, orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.grid(row=23, column=1, columnspan=3, sticky="ew")

        self.progress_bar["maximum"] = total_files
        self.progress_bar["value"] = 0

    def update_progress_bar_image_processor(self, total_files: int, i: int):
        self.progress_bar["value"] = i
        self.percent.set(
            f"{i}/{total_files} images processed ({int((i/total_files)*100)}%)"
        )
        self.images_tab.update()

    def finish_progress_bar_image_processor(self, execution_time: float):
        self.percent.set(f"Done! Processed all images in {execution_time} seconds")
        self.images_tab.update()
        self.progress_bar.pack_forget()

    def create_progress_bar_video_processor(self, total_files: int):
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
        self.progress_bar["value"] = i
        self.percent.set(
            f"{i}/{total_files} videos processed ({int((i/total_files)*100)}%), see the command window for the estimated times..."
        )
        self.images_tab.update()

    def finish_progress_bar_video_processor(self, execution_time: float):
        self.percent.set(f"Done! Processed all videos in {execution_time} seconds")
        self.images_tab.update()
        self.progress_bar.pack_forget()

    @staticmethod
    def load_settings() -> dict:
        """
        Loads the settings from settings.json

        :return: dictionary with all the settings from settings.json
        """

        with open(SETTINGS_PATH, "r") as file:
            settings = json.load(file)
        return settings

    @staticmethod
    def save_settings(settings: dict):
        """
        Save the provided settings to the specified JSON settings file.

        :param settings: A dictionary containing the settings to be saved.
        """

        with open(SETTINGS_PATH, "w") as file:
            json.dump(settings, file, indent=4)


def create_settings_json():
    default_settings_json_template = {
        "image_processor": {
            "input_folder_path": "",
            "output_folder_path": "",
            "number_pixels": "",
            "add_logo": False,
            "logo_image_path": "",
            "scale": "10",
            "width_offset": "10",
            "height_offset": "10",
            "logo_corner": "Bottom Left",
        },
        "video_processor": {
            "input_folder_path": "",
            "output_folder_path": "",
            "logo_image_path": "",
            "scale": "13",
            "width_offset": "10",
            "height_offset": "15",
            "logo_corner": "Bottom Left",
        },
    }

    with open(SETTINGS_PATH, "w") as file:
        json.dump(default_settings_json_template, file, indent=4)

    print("Created settings.json")


if __name__ == "__main__":
    if not os.path.exists(SETTINGS_PATH):
        create_settings_json()

    main_window = MainWindow()
    main_window.run()
    main_window.mainloop()
