import ex2_1 as gradient
import matplotlib.pyplot as plt
import numpy as np


def read_file(file: str):
    with open(file, 'r') as f:
        width, height, cell_size_cm = map(int, f.readline().split())
        data = f.readlines()
    data_arr = np.genfromtxt(data, dtype=float)
    return width, height, cell_size_cm, data_arr


def adjust_brightness(color, factor):
    adjusted_color = (
        int(min(color[0] * factor, 255)),  # Red channel
        int(min(color[1] * factor, 255)),  # Green channel
        int(min(color[2] * factor, 255))   # Blue channel
    )

    return adjusted_color

def apply_gradient(data: np.array, height: int, width: int):
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    max_val = data.max()
    for i in range(height):
        for j in range(width):
            v = round(data[i, j] / max_val, 2)    # Normalization to values between 0 and 1
            r, g, b = gradient.gradient_rgb_wb_custom(v)
            rgb_image[i, j] = (int(r * 255), int(g * 255), int(b * 255))
            diff = 0
            if j > 0:
                diff = data[i, j] - data[i, j - 1]
            diff = round(7 * diff / max_val)
            if diff > 0:
                rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 - diff)
            else:
                rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 - diff)
    return rgb_image


def make_map(data: np.array, height: int, width: int):
    fig = plt.figure()
    rgb_image = apply_gradient(data, height, width)
    plt.imshow(rgb_image)
    plt.title('Topological map')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('map.pdf')
    # plt.show()


if __name__ == '__main__':
    dem_file = '../data/lab2_data/big.dem'
    w, h, c, d = read_file(dem_file)
    print(f"Map width: {w}\nMap height: {h}\nDistance between map points: {c / 100}[m]\n")
    print(d.shape)
    print(d.max())
    make_map(d, h, w)
    print("Map created")
