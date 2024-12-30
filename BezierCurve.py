import matplotlib.pyplot as plt

P0 = (0, 0)
P1 = (1, 3)
P2 = (3, -3)
P3 = (4, 0)

num_points = 40
points = []

for i in range(num_points):
    t = i / (num_points - 1)
    x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
    y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
    points.append((x, y))

x_coords, y_coords = zip(*points)

plt.plot(x_coords, y_coords, marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cubic Bézier Curve')
plt.grid(True)
plt.show()
