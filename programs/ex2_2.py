import ex2_1 as gradient
import matplotlib.pyplot as plt
import numpy as np
import math as m


# Function reading the file
def read_file(file: str):
    with open(file, 'r') as f:
        width, height, cell_size_cm = map(int, f.readline().split())
        data = f.readlines()
    data_arr = np.genfromtxt(data, dtype=float)
    return width, height, cell_size_cm, data_arr


# Function for rgb color brightness adjustment
def adjust_brightness(color, factor):
    # Factor > 1 -> brighter | Factor < 1 -> darker
    adjusted_color = (
        int(min(color[0] * factor, 255)),  # Red channel
        int(min(color[1] * factor, 255)),  # Green channel
        int(min(color[2] * factor, 255))   # Blue channel
    )

    return adjusted_color


# Coloring function
def apply_gradient(data: np.array, height: int, width: int, dist: int):
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    angle_image = np.zeros((height, width), dtype=np.uint8)
    max_val = data.max()
    sun_vector = np.array([-dist, 100, -dist])
    for i in range(height):
        for j in range(width):
            v = round(data[i, j] / max_val, 2)    # Normalization to values between 0 and 1
            r, g, b = gradient.gradient_rgb_wb_custom(v)
            rgb_image[i, j] = (int(r * 255), int(g * 255), int(b * 255))
            # Central left and upper "triangle" squares as in instructions       # -> upper
            #                                                           left <- ## -> centre
            centre = np.array([i * dist, data[i, j], j * dist])
            if 0 < i < height and 0 < j < width:
                left = np.array([(i - 1) * dist, data[i - 1, j], j * dist])
                upper = np.array([i * dist, data[i][j - 1], (j - 1) * dist])
                centre_sun = sun_vector - centre                                # Sun-central point vector
                perpendicular_v = np.cross(left - centre, upper - centre)       # Vector perpendicular to triangle
                # Calculating the angle of slope
                angle = m.degrees(np.arccos(
                    np.clip(
                        np.dot(perpendicular_v, centre_sun) / (
                                np.linalg.norm(perpendicular_v) * np.linalg.norm(centre_sun)), -1, 1)))
                angle_image[i, j] = angle
    # Shading loop
    for i in range(height):
        for j in range(width):
            if 0 < i < height and 0 < j < width:
                # Checking if neighboring pixel's slope is higher or lower
                if angle_image[i - 1, j] < angle_image[i, j]:
                    rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 - 0.15)
                else:
                    rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 + 0.15)
                if angle_image[i, j - 1] < angle_image[i, j]:
                    rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 - 0.15)
                else:
                    rgb_image[i, j] = adjust_brightness(rgb_image[i, j], 1 + 0.15)

    return rgb_image


# Function drawing the map
def make_map(data: np.array, height: int, width: int, dist: int):
    fig = plt.figure()
    rgb_image = apply_gradient(data, height, width, dist)
    plt.imshow(rgb_image)
    plt.title('Topological map')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('map.pdf')
    # plt.show()


if __name__ == '__main__':
    dem_file = '../data/lab2_data/big.dem'
    w, h, c, d = read_file(dem_file)
    print(f"Map width: {w}\nMap height: {h}\nDistance between map points: {c / 100}[m]")
    make_map(d, h, w, c)
    print("Map created!")
