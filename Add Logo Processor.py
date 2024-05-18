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

global main_window  # main_window gets defined if __name__ == "__main__"

if getattr(sys, "frozen", False):
    # If the application is run as a bundled executable, use the directory of the executable
    MAIN_PATH = os.path.dirname(sys.executable)
else:
    # Otherwise, just use the normal directory where the script resides
    MAIN_PATH = os.path.abspath(os.path.dirname(__file__))

SETTINGS_PATH = Path(MAIN_PATH, "settings.json")

# TODO: Add a togglable function which can convert images to png format
# TODO: Make duplicate elements in gui dynamic by giving them the ability to be displayed in both tabs while keeping their different functions in their individual tabs (?v1.0x)

# class ImageProcessor:

#     @staticmethod
#     def resize_image_operation(
#         input_folder,
#         output_folder,
#         file,
#         min_pixels,
#         overwrite_output_images,
#         logo_img,
#         scale,
#         offset_logo_width,
#         offset_logo_height,
#         selected_corner,
#     ):
#         output_file = os.path.join(output_folder, file)

#         if not os.path.exists(output_file) or overwrite_output_images:
#             # Open the input image
#             input_file = os.path.join(input_folder, file)
#             img = cv2.imread(input_file)
#             height, width, _ = img.shape

#             # Calculate the new dimensions for resizing
#             if width < height:
#                 new_width = min_pixels
#                 new_height = int(height * (min_pixels / width))
#             else:
#                 new_height = min_pixels
#                 new_width = int(width * (min_pixels / height))
#             img = cv2.resize(img, (new_width, new_height))

#             if self.toggle_logo.get() == 1:
#                 logo_height, logo_width, _ = logo_img.shape
#                 logo_aspectratio = logo_width / logo_height
#                 scale_factor = scale / 100

#                 if width < height:
#                     logo_height = new_height * scale_factor
#                     logo_width = (
#                         logo_height * logo_aspectratio
#                         if logo_aspectratio != 1
#                         else logo_height
#                     )
#                 else:
#                     logo_width = new_width * scale_factor
#                     logo_height = (
#                         logo_width / logo_aspectratio
#                         if logo_aspectratio != 1
#                         else logo_width
#                     )

#                 logo_img = cv2.resize(logo_img, (int(logo_width), int(logo_height)))

#                 offset_logo_width = (offset_logo_width / 100) * (
#                     new_width * 0.5
#                 )  # 0.5 is a reducing factor to allow more accurate control over offset_logo_width
#                 offset_logo_height = (offset_logo_height / 100) * (
#                     new_height * 0.5
#                 )  # 0.5 is a reducing factor to allow more accurate control over offset_logo_height

#                 if selected_corner == "Top Left":
#                     position = int(0 + offset_logo_width), int(0 + offset_logo_height)
#                 elif selected_corner == "Top Right":
#                     position = int(new_width - logo_width - offset_logo_width), int(
#                         0 + offset_logo_height
#                     )
#                 elif selected_corner == "Bottom Left":
#                     position = int(0 + offset_logo_width), int(
#                         new_height - logo_height - offset_logo_height
#                     )
#                 elif selected_corner == "Bottom Right":
#                     position = int(new_width - logo_width - offset_logo_width), int(
#                         new_height - logo_height - offset_logo_height
#                     )

#                 x_offset, y_offset = position
#                 y1, y2 = y_offset, y_offset + logo_img.shape[0]
#                 x1, x2 = x_offset, x_offset + logo_img.shape[1]

#                 alpha_s = logo_img[:, :, 3] / 255.0
#                 alpha_l = 1.0 - alpha_s

#                 for c in range(0, 3):
#                     img[y1:y2, x1:x2, c] = (
#                         alpha_s * logo_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c]
#                     )

#             cv2.imwrite(output_file, img)

#     # TODO: Rebuild all save file logic
#     @staticmethod
#     def resize_images(
#         input_path: str,
#         output_path: str,
#         min_pixels: int,
#         overwrite_output_images: bool,
#         settings: str,
#     ):
#         start_time = time.time()
#         input_folder = os.path.join(entryinput.get())
#         output_folder = os.path.join(entryoutput.get())

#         min_pixels = int(entrynumpix.get())

#         overwrite_output_images = bool(checkboxoverwrite.get())

#         # Check if the min_pixels value hasn't changed since the last resize operation, otherwise it will overwrite all images in the output folder
#         with open(settings, "r") as file:
#             lines = file.readlines()
#             if len(lines) >= 3:
#                 third_line = lines[2].strip()
#                 if not str(min_pixels) == third_line:
#                     overwrite_output_images = True

#         files = os.listdir(input_folder)
#         files = sorted(
#             files,
#             key=lambda f: os.path.getsize(os.path.join(input_folder, f)),
#             reverse=True,
#         )

#         logo_original_path = entrylogo.get()
#         logo_img = cv2.imread(logo_original_path, cv2.IMREAD_UNCHANGED)
#         scale = int(entryscalelogo.get())
#         offset_logo_width = int(entryoffsetwidthlogo.get())
#         offset_logo_height = int(entryoffsetheightlogo.get())
#         selected_corner = selecting_corner.get()

#         # Write settings to the settings file
#         with open(settings, "w") as file:
#             file.write(
#                 f"{entryinput.get()}\n"
#                 f"{entryoutput.get()}\n"
#                 f"{entrynumpix.get()}\n"
#                 f"{togglelogo.get()}\n"
#                 f"{entrylogo.get()}\n"
#                 f"{entryscalelogo.get()}\n"
#                 f"{entryoffsetwidthlogo.get()}\n"
#                 f"{entryoffsetheightlogo.get()}\n"
#                 f"{selected_corner}"
#             )

#         progress_bar = ttk.Progressbar(
#             master=images_tab, orient="horizontal", length=400, mode="determinate"
#         )
#         progress_bar.grid(row=23, column=1, columnspan=3, sticky="ew")

#         total_files = len(files)
#         progress_bar["maximum"] = total_files
#         progress_bar["value"] = 0

#         with ThreadPoolExecutor() as executor:
#             futures = []
#             for i, file in enumerate(files, start=1):
#                 future = executor.submit(
#                     resize_image_operation,
#                     input_folder,
#                     output_folder,
#                     file,
#                     min_pixels,
#                     overwrite_output_images,
#                     logo_img,
#                     scale,
#                     offset_logo_width,
#                     offset_logo_height,
#                     selected_corner,
#                 )
#                 futures.append(future)

#             for i, future in enumerate(as_completed(futures), start=1):
#                 future.result()  # Wait for the resize_image operation to complete
#                 progress_bar["value"] = i
#                 percent.set(
#                     f"{i}/{total_files} images processed ({int((i/total_files)*100)}%)"
#                 )
#                 root.update()

#         end_time = time.time()
#         execution_time = round(end_time - start_time, 3)
#         print(f"All images are processed in {execution_time} seconds")
#         percent.set(f"Done! Processed all images in {execution_time} seconds")
#         root.update()
#         progress_bar.pack_forget()
#         time.sleep(5)
#         root.destroy()


class VideoProcessor:
    def __init__(self):
        pass


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

        images_tab = tabview.add("Images")
        videos_tab = tabview.add("Videos")

        ###############################
        # 	    Image Processor	      #
        ###############################

        # Define grid
        images_tab.columnconfigure(1, weight=1)
        images_tab.columnconfigure(2, weight=1)
        images_tab.columnconfigure(3, weight=1)

        for i in range(24):
            images_tab.rowconfigure(i, weight=1)  # Create row 1 to 20

        # Input folder
        label_input = ctk.CTkLabel(images_tab, text="Input Folder", font=("Arial", 14))
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_image = ctk.CTkEntry(
            images_tab, placeholder_text="Path of input folder"
        )
        self.entry_input_image.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            images_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_image),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(
            images_tab, text="Output Folder", font=("Arial", 14)
        )
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_image = ctk.CTkEntry(
            images_tab, placeholder_text="Path of output folder"
        )
        self.entry_output_image.grid(row=5, column=1, columnspan=3, sticky="ew")

        browse_button_output = ctk.CTkButton(
            images_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(self.entry_output_image),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Number of pixels
        label_num_pixels = ctk.CTkLabel(
            images_tab,
            text="Maximum number of pixels of the smallest side",
            font=("Arial", 14),
        )
        label_num_pixels.grid(row=7, column=1, columnspan=3, sticky="s")

        self.entry_num_pixels = ctk.CTkEntry(
            images_tab, placeholder_text="Number of pixels"
        )
        self.entry_num_pixels.grid(row=8, column=1, columnspan=3)

        # Logo image
        self.frame_logo = ctk.CTkFrame(
            images_tab,
            border_width=3,
            border_color="dark blue",
            fg_color="transparent",
        )
        self.frame_logo.grid(row=10, rowspan=6, column=1, columnspan=3, sticky="nsew")

        self.toggle_logo = ctk.CTkSwitch(
            images_tab,
            text="Add Logo to Image",
            font=("Arial", 18),
            command=lambda: self.update_logo_border_color(),
        )
        self.toggle_logo.grid(row=10, column=1, columnspan=3)

        label_logo = ctk.CTkLabel(
            images_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        label_logo.grid(row=11, column=2, sticky="s")

        self.entry_logo_image = ctk.CTkEntry(
            images_tab, placeholder_text="Path to Logo Image"
        )
        self.entry_logo_image.grid(row=12, column=2, sticky="ew")

        browse_button_input = ctk.CTkButton(
            images_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_image),
        )
        browse_button_input.grid(row=13, column=2, sticky="n")

        label_scale_logo = ctk.CTkLabel(
            images_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 10]",
            font=("Arial", 14),
        )
        label_scale_logo.grid(row=14, column=2, sticky="s")

        self.entry_scale_logo_image = ctk.CTkEntry(images_tab, placeholder_text="Scale")
        self.entry_scale_logo_image.grid(row=15, column=2, sticky="n")

        # Logo corner selection
        label_corner_selection = ctk.CTkLabel(
            images_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_corner_selection.grid(row=10, column=3, sticky="s")

        images_tab_cornerselection = ctk.CTkFrame(images_tab)
        images_tab_cornerselection.grid(row=11, column=3, sticky="ns")

        images_tab_cornerselection.columnconfigure((1, 2), weight=1)
        images_tab_cornerselection.rowconfigure((1, 2), weight=1)

        selecting_corner = ctk.StringVar()

        self.radiobutton_topleft_image = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Top Left",
            variable=selecting_corner,
            value="Top Left",
        )
        self.radiobutton_topleft_image.grid(row=1, column=1)

        self.radiobutton_topright_image = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Top Right",
            variable=selecting_corner,
            value="Top Right",
        )
        self.radiobutton_topright_image.grid(row=1, column=2, sticky="e")

        self.radiobutton_bottomleft_image = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Bottom Left",
            variable=selecting_corner,
            value="Bottom Left",
        )
        self.radiobutton_bottomleft_image.grid(row=2, column=1)

        self.radiobutton_bottomright_image = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Bottom Right",
            variable=selecting_corner,
            value="Bottom Right",
        )
        self.radiobutton_bottomright_image.grid(row=2, column=2, sticky="e")

        # Offset logo image
        label_offset_width_logo = ctk.CTkLabel(
            images_tab,
            text="Width Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_width_logo.grid(row=12, column=3, sticky="s")

        self.entry_offset_width_logo_image = ctk.CTkEntry(
            images_tab, placeholder_text="Width Offset"
        )
        self.entry_offset_width_logo_image.grid(row=13, column=3, sticky="n")

        label_offset_height_logo = ctk.CTkLabel(
            images_tab,
            text="Height Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_height_logo.grid(row=14, column=3, sticky="s")

        self.entry_offset_height_logo_image = ctk.CTkEntry(
            images_tab, placeholder_text="Height Offset"
        )
        self.entry_offset_height_logo_image.grid(row=15, column=3, sticky="n")

        # Overwrite output images
        checkbox_overwrite = ctk.CTkCheckBox(
            images_tab,
            text="Overwrite existing images in the output folder",
            font=("Arial", 14),
        )
        checkbox_overwrite.grid(row=17, column=1, columnspan=3)

        # Resize button
        button = ctk.CTkButton(
            images_tab,
            text="Resize",
            font=("Arial", 16),
            command=self.check_values_and_paths,
        )
        button.grid(row=19, column=1, columnspan=3, sticky="ns")

        # Progress bar
        percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(
            images_tab, textvariable=percent, font=("Arial", 15)
        )
        label_percent.grid(row=21, column=1, columnspan=3, sticky="s")

        ###############################
        # 	    Video Processor	      #
        ###############################

        # Define grid
        videos_tab.columnconfigure(1, weight=1)
        videos_tab.columnconfigure(2, weight=1)
        videos_tab.columnconfigure(3, weight=1)

        for i in range(20):
            videos_tab.rowconfigure(i, weight=1)

        # Input folder
        label_input = ctk.CTkLabel(
            master=videos_tab, text="Input Folder", font=("Arial", 14)
        )
        label_input.grid(row=1, column=1, columnspan=3, sticky="s")

        self.entry_input_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path of input folder"
        )
        self.entry_input_video.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=videos_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(self.entry_input_video),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        label_output = ctk.CTkLabel(
            master=videos_tab, text="Output Folder", font=("Arial", 14)
        )
        label_output.grid(row=4, column=1, columnspan=3, sticky="s")

        self.entry_output_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path of output folder"
        )
        self.entry_output_video.grid(row=5, column=1, columnspan=3, sticky="ew")

        browse_button_output = ctk.CTkButton(
            master=videos_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(self.entry_output_video),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Logo image
        label_logo_title = ctk.CTkLabel(
            master=videos_tab, text="Logo parameters", font=("Arial", 20)
        )
        label_logo_title.grid(row=8, column=1, columnspan=3)

        label_logo = ctk.CTkLabel(
            master=videos_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        label_logo.grid(row=9, column=2, sticky="s")

        self.entry_logo_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path to Logo Image"
        )
        self.entry_logo_video.grid(row=10, column=2, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=videos_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(self.entry_logo_video),
        )
        browse_button_input.grid(row=11, column=2, sticky="n")

        label_scale_logo = ctk.CTkLabel(
            master=videos_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 13]",
            font=("Arial", 14),
        )
        label_scale_logo.grid(row=12, column=2, sticky="s")

        self.entry_scale_logo_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Scale"
        )
        self.entry_scale_logo_video.grid(row=13, column=2, sticky="n")

        # Logo corner selection
        label_corner_selection = ctk.CTkLabel(
            master=videos_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_corner_selection.grid(row=8, column=3, sticky="s")

        frame_corner_selection = ctk.CTkFrame(master=videos_tab)
        frame_corner_selection.grid(row=9, column=3, sticky="ns")

        frame_corner_selection.columnconfigure((1, 2), weight=1)
        frame_corner_selection.rowconfigure((1, 2), weight=1)

        selecting_corner = ctk.StringVar()

        self.radiobutton_topleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Top Left",
            variable=selecting_corner,
            value="Top Left",
        )
        self.radiobutton_topleft_video.grid(row=1, column=1)

        self.radiobutton_topright_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Top Right",
            variable=selecting_corner,
            value="Top Right",
        )
        self.radiobutton_topright_video.grid(row=1, column=2, sticky="e")

        self.radiobutton_bottomleft_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Bottom Left",
            variable=selecting_corner,
            value="Bottom Left",
        )
        self.radiobutton_bottomleft_video.grid(row=2, column=1)

        self.radiobutton_bottomright_video = ctk.CTkRadioButton(
            master=frame_corner_selection,
            text="Bottom Right",
            variable=selecting_corner,
            value="Bottom Right",
        )
        self.radiobutton_bottomright_video.grid(row=2, column=2, sticky="e")

        # Offset logo image
        label_offset_width_logo = ctk.CTkLabel(
            master=videos_tab,
            text="Width Offset [Range: -100 to 100, Default: 10]",
            font=("Arial", 14),
        )
        label_offset_width_logo.grid(row=10, column=3, sticky="s")

        self.entry_offset_width_logo_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Width Offset"
        )
        self.entry_offset_width_logo_video.grid(row=11, column=3, sticky="n")

        label_offset_height_logo = ctk.CTkLabel(
            master=videos_tab,
            text="Height Offset [Range: -100 to 100, Default: 15]",
            font=("Arial", 14),
        )
        label_offset_height_logo.grid(row=12, column=3, sticky="s")

        self.entry_offset_height_logo_video = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Height Offset"
        )
        self.entry_offset_height_logo_video.grid(row=13, column=3, sticky="n")

        # Overwrite output images
        checkbox_overwrite = ctk.CTkCheckBox(
            master=videos_tab,
            text="Overwrite existing videos in the output folder",
            font=("Arial", 14),
        )
        checkbox_overwrite.grid(row=15, column=1, columnspan=3)

        # Add Logo to Videos button
        button = ctk.CTkButton(
            master=videos_tab,
            text="Add Logo to Videos",
            font=("Arial", 16),
            command=self.check_values_and_paths,
        )
        button.grid(row=17, column=1, columnspan=3, sticky="ns")

        # Progress bar
        percent = ctk.StringVar()

        label_percent = ctk.CTkLabel(
            master=videos_tab, textvariable=percent, font=("Arial", 15)
        )
        label_percent.grid(row=19, column=1, columnspan=3, sticky="s")

    def browse_path(self, entry):
        pass

    def check_values_and_paths(self):
        input_folder = self.entry_input.get()
        output_folder = self.entry_output.get()
        logo_image = self.entry_logo.get()

        if not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Invalid path for input folder")
            return
        if not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Invalid path for output folder")
            return
        if not os.path.isfile(logo_image):
            messagebox.showerror(
                "Error", "Invalid path for logo image: No file recognized"
            )
            return
        scale_value = int(self.entry_scale_logo.get())
        width_offset = int(self.entry_offset_width_logo.get())
        height_offset = int(self.entry_offset_height_logo.get())

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
            pixels = int(self.entry_num_pixels.get())
            if pixels <= 0:
                raise ValueError("Invalid pixel value")
        except ValueError:
            messagebox.showerror("Error", "Invalid pixel value")
            return

        self.save_settings()
        # XXX:ImageProcessor.resize_images()

    def update_framelogo_border_color(self):
        if self.toggle_logo.get() == 1:
            self.frame_logo.configure(border_color="dark blue")
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
