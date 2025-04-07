from PIL import Image
import numpy as np
import math
import os


class image_processor:
  def __init__(self, i):
    self.path = i
    self.img = Image.open(i)
    self.img_G = Image.open(i)

  def get_size(self, *args):
    return next((getattr(self, arg).size for arg in args if hasattr(self, arg)), self.img.size)

  def sobel(self):
    self.img_L = self.img.convert('L')

    s_x: np.ndarray = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
    s_y: np.ndarray = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    g_ = (lambda mat: sum(mat[i][j] * self.img_L.getpixel((x + j, y + i)) for i in range(3) for j in range(3)))

    for x in range(self.get_size("img_L")[0] - 2):
      for y in range(self.get_size("img_L")[1] - 2):
        g_x: float = g_(s_x)
        g_y: float = g_(s_y)
        G: int = int(math.sqrt((g_x**2) + (g_y**2)))
        self.img_G.putpixel((x, y), (G, G, G))

    self.img_G.save(f"./{self.path.split(".")[0]}.bmp", format="BMP", bitmap_format="bmp")

  def segment(self, threshold):
    class params:
      nearest_neighbors = (lambda x, y: [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)])
      colored = (lambda r, g, b, a: True if any(threshold < abs(255 - v) for v in [r, g, b, a]) else False)

    w, h = self.img.size
    print(params.nearest_neighbors(1, 1))
    search_mat = [[]]
    for x in range(w - 2):
      for y in range(h - 2):
        r, g, b, a = self.img_G.getpixel((x, y))
        # TODO folding image for edge detection


if __name__ == "__main__":
  img: image_processor = image_processor("data/example_1.png")
  # img.sobel()
  # img.segment(2)
