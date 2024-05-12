import cv2
from djitellopy import Tello
from ultralytics import YOLO
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np

class DroneDetectionApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()

        self.custom_model_path = "../pythonProject/weeddect.pt"
        self.model = YOLO(self.custom_model_path)

        self.classNames = [
            "Broadleaf Weed", "Carpet Weed", "Crabgrass Weed", "Eclipta Weed",
            "Goosegrass Weed", "Morningglory Weed",
            "Nutsedge Weed", "Palmer Amaranth Weed", "Prickly Sida Weed",
            "Purslane Weed", "Ragweed Weed", "Sicklepod Weed",
            "SpottedSpurge Weed", "Spurred Anoda Weed", "Swinecress Weed", "Waterhemp Weed"
        ]

        self.class_colors = {
            "Broadleaf Weed": (0, 0, 255),  # Red
            "Carpet Weed": (0, 255, 0),  # Green
            "Crabgrass Weed": (255, 0, 0),  # Blue
            "Eclipta Weed": (255, 255, 0),  # Cyan
            "Goosegrass Weed": (0, 255, 255),  # Yellow
            "Morningglory Weed": (255, 0, 255),  # Magenta
            "Nutsedge Weed": (255, 255, 255),  # White
            "Palmer Amaranth Weed": (128, 0, 128),  # Purple
            "Prickly Sida Weed": (128, 128, 0),  # Olive
            "Purslane Weed": (0, 128, 128),  # Teal
            "Ragweed Weed": (0, 64, 128),  # Navy
            "Sicklepod Weed": (128, 64, 0),  # Maroon
            "SpottedSpurge Weed": (64, 128, 0),  # Lime
            "Spurred Anoda Weed": (128, 0, 64),  # Fuchsia
            "Swinecress Weed": (0, 128, 64),  # Aqua
            "Waterhemp Weed": (255, 165, 0)  # Orange
        }

    def build(self):
        self.img = Image()
        Clock.schedule_interval(self.update, 1.0/30.0)  # Update at 30 fps
        return self.img

    def update(self, dt):
        frame = self.drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self.model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = self.classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = self.class_colors.get(class_name, (0, 0, 255))  # Default: Red
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Convert the frame to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

if __name__ == '__main__':
    DroneDetectionApp().run()
