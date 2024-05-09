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


class ImageProcessor:
    def __init__(self):
        pass


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
        labelinput = ctk.CTkLabel(images_tab, text="Input Folder", font=("Arial", 14))
        labelinput.grid(row=1, column=1, columnspan=3, sticky="s")

        entryinput = ctk.CTkEntry(images_tab, placeholder_text="Path of input folder")
        entryinput.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            images_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(entryinput),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        labeloutput = ctk.CTkLabel(images_tab, text="Output Folder", font=("Arial", 14))
        labeloutput.grid(row=4, column=1, columnspan=3, sticky="s")

        entryoutput = ctk.CTkEntry(images_tab, placeholder_text="Path of output folder")
        entryoutput.grid(row=5, column=1, columnspan=3, sticky="ew")

        browse_button_output = ctk.CTkButton(
            images_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(entryoutput),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Number of pixels
        labelnumpix = ctk.CTkLabel(
            images_tab,
            text="Maximum number of pixels of the smallest side",
            font=("Arial", 14),
        )
        labelnumpix.grid(row=7, column=1, columnspan=3, sticky="s")

        entrynumpix = ctk.CTkEntry(images_tab, placeholder_text="Number of pixels")
        entrynumpix.grid(row=8, column=1, columnspan=3)

        # Logo image
        images_tab_logo = ctk.CTkFrame(
            images_tab,
            border_width=3,
            border_color="dark blue",
            fg_color="transparent",
        )
        images_tab_logo.grid(row=10, rowspan=6, column=1, columnspan=3, sticky="nsew")

        togglelogo = ctk.CTkSwitch(
            images_tab,
            text="Add Logo to Image",
            font=("Arial", 18),
            command=lambda: self.update_logo_border_color(),
        )
        togglelogo.grid(row=10, column=1, columnspan=3)

        labellogo = ctk.CTkLabel(
            images_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        labellogo.grid(row=11, column=2, sticky="s")

        entrylogo = ctk.CTkEntry(images_tab, placeholder_text="Path to Logo Image")
        entrylogo.grid(row=12, column=2, sticky="ew")

        browse_button_input = ctk.CTkButton(
            images_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(entrylogo),
        )
        browse_button_input.grid(row=13, column=2, sticky="n")

        labelscalelogo = ctk.CTkLabel(
            images_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 10]",
            font=("Arial", 14),
        )
        labelscalelogo.grid(row=14, column=2, sticky="s")

        entryscalelogo = ctk.CTkEntry(images_tab, placeholder_text="Scale")
        entryscalelogo.grid(row=15, column=2, sticky="n")

        # Logo corner selection
        label_cornerselection = ctk.CTkLabel(
            images_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_cornerselection.grid(row=10, column=3, sticky="s")

        images_tab_cornerselection = ctk.CTkFrame(images_tab)
        images_tab_cornerselection.grid(row=11, column=3, sticky="ns")

        images_tab_cornerselection.columnconfigure((1, 2), weight=1)
        images_tab_cornerselection.rowconfigure((1, 2), weight=1)

        selecting_corner = ctk.StringVar()

        radiobutton_topleft = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Top Left",
            variable=selecting_corner,
            value="Top Left",
        )
        radiobutton_topleft.grid(row=1, column=1)

        radiobutton_topright = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Top Right",
            variable=selecting_corner,
            value="Top Right",
        )
        radiobutton_topright.grid(row=1, column=2, sticky="e")

        radiobutton_bottomleft = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Bottom Left",
            variable=selecting_corner,
            value="Bottom Left",
        )
        radiobutton_bottomleft.grid(row=2, column=1)

        radiobutton_bottomright = ctk.CTkRadioButton(
            images_tab_cornerselection,
            text="Bottom Right",
            variable=selecting_corner,
            value="Bottom Right",
        )
        radiobutton_bottomright.grid(row=2, column=2, sticky="e")

        # Offset logo image
        labeloffsetwidthlogo = ctk.CTkLabel(
            images_tab,
            text="Width Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        labeloffsetwidthlogo.grid(row=12, column=3, sticky="s")

        entryoffsetwidthlogo = ctk.CTkEntry(images_tab, placeholder_text="Width Offset")
        entryoffsetwidthlogo.grid(row=13, column=3, sticky="n")

        labeloffsetheightlogo = ctk.CTkLabel(
            images_tab,
            text="Height Offset [Range: 0 to 100, Default: 10]",
            font=("Arial", 14),
        )
        labeloffsetheightlogo.grid(row=14, column=3, sticky="s")

        entryoffsetheightlogo = ctk.CTkEntry(
            images_tab, placeholder_text="Height Offset"
        )
        entryoffsetheightlogo.grid(row=15, column=3, sticky="n")

        # Overwrite output images
        checkboxoverwrite = ctk.CTkCheckBox(
            images_tab,
            text="Overwrite existing images in the output folder",
            font=("Arial", 14),
        )
        checkboxoverwrite.grid(row=17, column=1, columnspan=3)

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
        labelinput = ctk.CTkLabel(
            master=videos_tab, text="Input Folder", font=("Arial", 14)
        )
        labelinput.grid(row=1, column=1, columnspan=3, sticky="s")

        entryinput = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path of input folder"
        )
        entryinput.grid(row=2, column=1, columnspan=3, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=videos_tab,
            text="Browse input folder",
            command=lambda: self.browse_path(entryinput),
        )
        browse_button_input.grid(row=3, column=1, columnspan=3, sticky="n")

        # Output folder
        labeloutput = ctk.CTkLabel(
            master=videos_tab, text="Output Folder", font=("Arial", 14)
        )
        labeloutput.grid(row=4, column=1, columnspan=3, sticky="s")

        entryoutput = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path of output folder"
        )
        entryoutput.grid(row=5, column=1, columnspan=3, sticky="ew")

        browse_button_output = ctk.CTkButton(
            master=videos_tab,
            text="Browse output folder",
            command=lambda: self.browse_path(entryoutput),
        )
        browse_button_output.grid(row=6, column=1, columnspan=3, sticky="n")

        # Logo image
        labellogotitle = ctk.CTkLabel(
            master=videos_tab, text="Logo parameters", font=("Arial", 20)
        )
        labellogotitle.grid(row=8, column=1, columnspan=3)

        labellogo = ctk.CTkLabel(
            master=videos_tab, text="Path to Logo Image", font=("Arial", 14)
        )
        labellogo.grid(row=9, column=2, sticky="s")

        entrylogo = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Path to Logo Image"
        )
        entrylogo.grid(row=10, column=2, sticky="ew")

        browse_button_input = ctk.CTkButton(
            master=videos_tab,
            text="Browse logo image",
            command=lambda: self.browse_path(entrylogo),
        )
        browse_button_input.grid(row=11, column=2, sticky="n")

        labelscalelogo = ctk.CTkLabel(
            master=videos_tab,
            text="Scale of the logo image [Range: 1 to 100 , Default: 13]",
            font=("Arial", 14),
        )
        labelscalelogo.grid(row=12, column=2, sticky="s")

        entryscalelogo = ctk.CTkEntry(master=videos_tab, placeholder_text="Scale")
        entryscalelogo.grid(row=13, column=2, sticky="n")

        # Logo corner selection
        label_cornerselection = ctk.CTkLabel(
            master=videos_tab, text="Corner Selection", font=("Arial", 14)
        )
        label_cornerselection.grid(row=8, column=3, sticky="s")

        frame_cornerselection = ctk.CTkFrame(master=videos_tab)
        frame_cornerselection.grid(row=9, column=3, sticky="ns")

        frame_cornerselection.columnconfigure((1, 2), weight=1)
        frame_cornerselection.rowconfigure((1, 2), weight=1)

        selecting_corner = ctk.StringVar()

        radiobutton_topleft = ctk.CTkRadioButton(
            master=frame_cornerselection,
            text="Top Left",
            variable=selecting_corner,
            value="Top Left",
        )
        radiobutton_topleft.grid(row=1, column=1)

        radiobutton_topright = ctk.CTkRadioButton(
            master=frame_cornerselection,
            text="Top Right",
            variable=selecting_corner,
            value="Top Right",
        )
        radiobutton_topright.grid(row=1, column=2, sticky="e")

        radiobutton_bottomleft = ctk.CTkRadioButton(
            master=frame_cornerselection,
            text="Bottom Left",
            variable=selecting_corner,
            value="Bottom Left",
        )
        radiobutton_bottomleft.grid(row=2, column=1)

        radiobutton_bottomright = ctk.CTkRadioButton(
            master=frame_cornerselection,
            text="Bottom Right",
            variable=selecting_corner,
            value="Bottom Right",
        )
        radiobutton_bottomright.grid(row=2, column=2, sticky="e")

        # Offset logo image
        labeloffsetwidthlogo = ctk.CTkLabel(
            master=videos_tab,
            text="Width Offset [Range: -100 to 100, Default: 10]",
            font=("Arial", 14),
        )
        labeloffsetwidthlogo.grid(row=10, column=3, sticky="s")

        entryoffsetwidthlogo = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Width Offset"
        )
        entryoffsetwidthlogo.grid(row=11, column=3, sticky="n")

        labeloffsetheightlogo = ctk.CTkLabel(
            master=videos_tab,
            text="Height Offset [Range: -100 to 100, Default: 15]",
            font=("Arial", 14),
        )
        labeloffsetheightlogo.grid(row=12, column=3, sticky="s")

        entryoffsetheightlogo = ctk.CTkEntry(
            master=videos_tab, placeholder_text="Height Offset"
        )
        entryoffsetheightlogo.grid(row=13, column=3, sticky="n")

        # Overwrite output images
        checkboxoverwrite = ctk.CTkCheckBox(
            master=videos_tab,
            text="Overwrite existing videos in the output folder",
            font=("Arial", 14),
        )
        checkboxoverwrite.grid(row=15, column=1, columnspan=3)

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

        labelpercent = ctk.CTkLabel(
            master=videos_tab, textvariable=percent, font=("Arial", 15)
        )
        labelpercent.grid(row=19, column=1, columnspan=3, sticky="s")

    def browse_path(self, entry):
        pass

    def check_values_and_paths(self):
        pass


def create_settings_json():
    default_settings_json_template = {"image_processor": {}, "video_processor": {}}

    with open(SETTINGS_PATH, "w") as file:
        json.dump({}, file)


if __name__ == "__main__":
    if os.path.exists(SETTINGS_PATH):
        create_settings_json()

    main_window = MainWindow()
    main_window.run()
    main_window.mainloop()
