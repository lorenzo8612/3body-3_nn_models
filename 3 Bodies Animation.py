raimport math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


##############################################################################################################

#File not needed to create the datasets for the neural networks

#############################################################################################################

class Body:
    def __init__(self, x, y, radius, color, mass, vx, vy):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.orbit = [(x, y)]


def update_position(body, dt):
    body.x += body.vx * dt
    body.y += body.vy * dt


def update_velocity(body, force, dt):
    ax = force[0] / body.mass
    ay = force[1] / body.mass
    body.vx += ax * dt
    body.vy += ay * dt


def gravitational_force(b1, b2, G=1.0, eps=1e-5):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist2 = dx*dx + dy*dy + eps*eps
    dist = math.sqrt(dist2)
    mag = G * b1.mass * b2.mass / dist2
    return (mag * dx/dist, mag * dy/dist)


def compute_forces(bodies):
    forces = [(0.0, 0.0) for _ in bodies]
    for i, b in enumerate(bodies):
        fx, fy = 0.0, 0.0
        for j, other in enumerate(bodies):
            if i != j:
                fx_i, fy_i = gravitational_force(b, other)
                fx += fx_i; fy += fy_i
        forces[i] = (fx, fy)
    return forces

def check_condition(bodies, max_distance_squared=200, min_distance_squared=0.001, eps=1e-5):
    body1 = bodies[0]
    body2 = bodies[1]
    body3 = bodies[2]
    dx = np.zeros(len(bodies))
    dy = np.zeros(len(bodies))
    dx[0] = body1.x - body2.x
    dy[0] = body1.y - body2.y

    dx[1] = body3.x - body2.x
    dy[1] = body3.y - body2.y

    dx[2] = body1.x - body3.x
    dy[2] = body1.y - body3.y

    distances_squared = dx ** 2 + dy ** 2 + eps * eps
    distance_squared_max = np.max(distances_squared)
    distance_squared_min = np.min(distances_squared)
    if distance_squared_max > max_distance_squared:
        label = 2
        condition = False
    elif distance_squared_min < min_distance_squared:
        label = 0
        condition = False
    else:
        condition = True
        label = 1
    return condition, label

#Initial conditions
Lmax = 1.5

r = np.random.uniform(0, 2 * np.pi, size=2)
L = Lmax * np.sqrt(np.random.uniform(0.7, 1, size=2))
x = L * np.cos(r)
y = L * np.sin(r)

body_A = Body(x[0], y[0], 0.1, "red", 2, 0.0, 0.0)
body_B = Body(-x[0], y[1], 0.1, "green", 2, 0.0, 0.0)
body_C = Body(-(2*x[0] - 2*x[0]), -(2*y[0] + 2*y[1]), 0.1, 'blue', 1, 0.0, 0.0)
bodies = [body_A, body_B, body_C]
# snapshot initial state
initial_state = [(b.x, b.y, b.vx, b.vy) for b in bodies]

dt = 0.01
max_frames = 1000

time_elapsed = 0.0




#Animation function
def animate(frame):
    global time_elapsed
    forces = compute_forces(bodies)
    for b, f in zip(bodies, forces):
        update_velocity(b, f, dt/2)
    for b in bodies:
        update_position(b, dt)
        b.orbit.append((b.x, b.y))
    forces = compute_forces(bodies)
    for b, f in zip(bodies, forces):
        update_velocity(b, f, dt/2)
    if not (check_condition(bodies))[0]:
        ani.event_source.stop()
        print("Stop!")

    # update clock
    time_elapsed += dt

    # redraw
    ax.cla()
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    # draw time in axes
    ax.text(0.02, 0.95, f"Time = {time_elapsed:.2f}", transform=ax.transAxes)
    for b in bodies:
        xs, ys = zip(*b.orbit)
        ax.plot(xs, ys, lw=1.5, color=b.color)
        ax.scatter(b.x, b.y, s=100, color=b.color)

#Set up figure and axes
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=range(max_frames), interval=20)

#Button callbacks:
def start(event):
    ani.event_source.start()

def stop(event):
    ani.event_source.stop()

def reset(event):
    global time_elapsed
    ani.event_source.stop()
    # restore initial state
    time_elapsed = 0.0
    for b, state in zip(bodies, initial_state):
        b.x, b.y, b.vx, b.vy = state
        b.orbit.clear()
        b.orbit.append((b.x, b.y))
    # redraw reset
    ax.cla()
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.text(0.02, 0.95, "Time = 0.00", transform=ax.transAxes)
    for b in bodies:
        ax.scatter(b.x, b.y, s=100, color=b.color)
    fig.canvas.draw()

#Add Buttons to the plot
ax_start = plt.axes([0.465, 0.9, 0.1, 0.05]) #x position, y position, length, height
ax_stop  = plt.axes([0.565,0.9, 0.1, 0.05])
ax_reset = plt.axes([0.365,0.9, 0.1, 0.05])
btn_start = Button(ax_start, 'Start')
btn_stop  = Button(ax_stop,  'Pause')
btn_reset = Button(ax_reset, 'Reset')
btn_start.on_clicked(start)
btn_stop.on_clicked(stop)
btn_reset.on_clicked(reset)

plt.show()
