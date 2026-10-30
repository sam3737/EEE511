import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):

    def __init__(self, size=1000, seed=0):
        self.rng = np.random.default_rng(seed)
        draw_functions = [self.draw_square, self.draw_circle, self.draw_triangle]
        imgs = np.stack([draw_functions[i % 3]() for i in range(size)])

        # Shape (N, 1, 128, 128) with images scaled to [0, 1] and masks strictly 0 or 1
        self.images = torch.tensor(imgs / 255, dtype=torch.float32).unsqueeze(1)
        self.masks = torch.tensor(imgs >= 200, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

    def generate_background(self):
        img = np.full((128, 128), self.rng.integers(0, 56), dtype=np.uint8)
        return img

    def generate_color(self):
        return int(self.rng.integers(200, 256))

    def draw_square(self):
        img = self.generate_background()
        color = self.generate_color()
        side_length = self.rng.integers(24, 51)
        top_left = self.rng.integers(0, 129 - side_length, size=2)
        cv2.rectangle(
            img, pt1=top_left, pt2=top_left + side_length - 1, color=color, thickness=-1
        )
        return img

    def draw_circle(self):
        img = self.generate_background()
        color = self.generate_color()
        radius = self.rng.integers(10, 26)
        center = self.rng.integers(radius, 128 - radius, size=2)
        cv2.circle(img, center=center, radius=radius, color=color, thickness=-1)
        return img

    def draw_triangle(self):
        img = self.generate_background()
        color = self.generate_color()
        lengths = self.rng.uniform(15, 30, 3)
        # Roughly evenly spaced angles so the vertices never bunch into a sliver
        angles = (
            self.rng.uniform(0, 2 * np.pi)
            + np.arange(3) * 2 * np.pi / 3
            + self.rng.uniform(-np.pi / 6, np.pi / 6, 3)
        )
        vertices = np.stack([np.cos(angles) * lengths, np.sin(angles) * lengths]).T
        vertices = np.round(vertices).astype(np.int32)
        vertices += self.rng.integers(
            -vertices.min(axis=0), 128 - vertices.max(axis=0), size=2, dtype=np.int32
        )
        cv2.fillPoly(img, [vertices], color=color)
        return img
