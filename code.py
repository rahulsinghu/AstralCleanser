import numpy as np
from skyfield.api import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# --- Load Satellite Data ---
stations_url = 'http://celestrak.org/NORAD/elements/active.txt'
satellites = load.tle_file(stations_url)
ts = load.timescale()

selected_sats = satellites[:5]
times = ts.utc(2025, 4, 19, range(0, 24*4))  # every 15 min

# --- Get satellite positions ---
paths = []
for sat in selected_sats:
    path = []
    for t in times:
        pos = sat.at(t).position.km
        path.append(pos)
    paths.append(np.array(path))  # convert to numpy array for easy slicing

# --- Detect close approaches (collisions) ---
threshold = 50  # km
collisions = []
for t in range(len(times)):
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = np.linalg.norm(paths[i][t] - paths[j][t])
            if d < threshold:
                collisions.append((t, i, j))

# --- Create 3D Plot ---
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green', 'orange', 'purple']

# --- Draw Earth Sphere ---
def draw_earth(radius=6371):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)

draw_earth()

# --- Setup plot limits ---
limit = 20000  # km
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("🛰️ Animated Satellite Orbits Around Earth 🌍", fontsize=14)

# --- Prepare plot elements for animation ---
lines = [ax.plot([], [], [], color=colors[i])[0] for i in range(len(paths))]
dots = [ax.plot([], [], [], marker='o', color=colors[i])[0] for i in range(len(paths))]
collision_dots = []

# --- Animate Frame ---
def animate(frame):
    for i, path in enumerate(paths):
        lines[i].set_data(path[:frame, 0], path[:frame, 1])
        lines[i].set_3d_properties(path[:frame, 2])
        dots[i].set_data(path[frame, 0], path[frame, 1])
        dots[i].set_3d_properties(path[frame, 2])

    # Show collisions
    for (t, i, j) in collisions:
        if t == frame:
            mid_point = (paths[i][frame] + paths[j][frame]) / 2
            c_dot = ax.plot([mid_point[0]], [mid_point[1]], [mid_point[2]],
                            marker='x', color='black', markersize=10)[0]
            collision_dots.append(c_dot)

    return lines + dots + collision_dots

# --- Start Animation ---
ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=100, blit=True)
plt.legend([f"Sat {i}" for i in range(len(paths))])
plt.tight_layout()
plt.show()

