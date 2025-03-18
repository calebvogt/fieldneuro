#to do:
# - drag method always outputs the same square, regardless of user input.
# - for multi point click method, a line should appear connecting each of the sequential user input clicks which produce a red dot. 
# - for multi point click method, the ffmpeg output fails. 

import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import subprocess

class VideoCropper:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        self.current_frame = 0
        self.rect_start = None
        self.rect_end = None
        self.polygon_points = []

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Crop Video")
        self.root.geometry("1100x700")
        self.root.state("zoomed")  # Maximize window

        # Create Canvas for video preview
        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=5)

        # Slider Variables
        self.frame_time = tk.DoubleVar(value=0)

        # Video Navigation
        tk.Label(self.root, text="Frame Time (seconds)").grid(row=1, column=0, columnspan=6)
        self.slider = tk.Scale(self.root, variable=self.frame_time, from_=0, to=self.duration, resolution=0.1, 
                               orient="horizontal", length=400, command=lambda x: self.update_preview())
        self.slider.grid(row=2, column=0, columnspan=6, padx=10, pady=5)

        # Time Adjustment Buttons
        button_spacing = {"padx": 2, "pady": 5}
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=6, pady=5)
        tk.Button(button_frame, text="-60s", command=lambda: self.adjust_time(-60)).pack(side="left", **button_spacing)
        tk.Button(button_frame, text="-30s", command=lambda: self.adjust_time(-30)).pack(side="left", **button_spacing)
        tk.Button(button_frame, text="-1s", command=lambda: self.adjust_time(-1)).pack(side="left", **button_spacing)
        tk.Button(button_frame, text="+1s", command=lambda: self.adjust_time(1)).pack(side="left", **button_spacing)
        tk.Button(button_frame, text="+30s", command=lambda: self.adjust_time(30)).pack(side="left", **button_spacing)
        tk.Button(button_frame, text="+60s", command=lambda: self.adjust_time(60)).pack(side="left", **button_spacing)

        # Crop Mode Buttons
        crop_frame = tk.Frame(self.root)
        crop_frame.grid(row=4, column=0, columnspan=6, pady=10)
        tk.Button(crop_frame, text="Crop via Rectangle (Drag)", command=self.start_rectangle_crop).pack(side="left", **button_spacing)
        tk.Button(crop_frame, text="Crop via Polygon (4 Clicks)", command=self.start_polygon_crop).pack(side="left", **button_spacing)

        # Final Crop Button
        self.crop_button = tk.Button(self.root, text="Crop Video", command=self.crop_video, state="disabled")
        self.crop_button.grid(row=5, column=2, columnspan=2, pady=20)

        self.update_preview()
        self.root.mainloop()

    def adjust_time(self, seconds):
        """Adjusts slider position by the given second count."""
        new_time = max(0, min(self.duration, self.frame_time.get() + seconds))
        self.frame_time.set(new_time)
        self.update_preview()

    def update_preview(self):
        """Updates the preview frame based on the selected time."""
        frame_idx = int(self.frame_time.get() * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.imencode('.png', frame)[1].tobytes()
            img = tk.PhotoImage(data=img)
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.image = img

    def start_rectangle_crop(self):
        """Activates rectangle cropping mode."""
        self.rect_start = None
        self.rect_end = None
        self.polygon_points = []
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.complete_rectangle)

    def start_rectangle(self, event):
        """Records the starting point of the rectangle."""
        self.rect_start = (event.x, event.y)

    def draw_rectangle(self, event):
        """Draws a rectangle as the user drags the mouse."""
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.rect_start[0], self.rect_start[1], event.x, event.y, outline="red", tag="rect")

    def complete_rectangle(self, event):
        """Finalizes the rectangle crop selection."""
        self.rect_end = (event.x, event.y)
        self.crop_button.config(state="normal")

    def start_polygon_crop(self):
        """Activates polygon cropping mode."""
        self.rect_start = None
        self.rect_end = None
        self.polygon_points = []
        self.canvas.bind("<ButtonPress-1>", self.add_polygon_point)

    def add_polygon_point(self, event):
        """Stores the clicked point for the polygon crop."""
        self.polygon_points.append((event.x, event.y))
        self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="red", tag="polygon")
        
        if len(self.polygon_points) == 4:
            self.crop_button.config(state="normal")

    def crop_video(self):
        """Executes FFmpeg to crop the video using the selected region."""
        base_name, ext = os.path.splitext(self.video_path)
        output_file = f"{base_name}_cropped.mp4"

        if self.rect_start and self.rect_end:
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            crop_cmd = f"ffmpeg -i \"{self.video_path}\" -vf \"crop={width}:{height}:{x1}:{y1}, pad=1080:1920:-1:-1:black\" \"{output_file}\""

        elif len(self.polygon_points) == 4:
            points = ":".join([f"{x}:{y}" for x, y in self.polygon_points])
            crop_cmd = f"ffmpeg -i \"{self.video_path}\" -vf \"crop={points}, pad=1080:1920:-1:-1:black\" \"{output_file}\""

        else:
            print("No valid crop area selected.")
            return

        print("Cropping video... Please wait.")
        subprocess.run(crop_cmd, shell=True)
        print(f"✅ Cropped video saved as: {output_file}")

        self.root.destroy()

def crop_video():
    """Prompts user to select a video and launches the cropping tool."""
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        VideoCropper(file_path)

if __name__ == "__main__":
    crop_video()
