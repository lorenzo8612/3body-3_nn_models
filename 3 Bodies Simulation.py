import math
from sre_constants import error
import random
import numpy as np
import torch


class Body:
    """Represents a body with methods to compute forces and displacements."""

    def __init__(self, x, y, radius, color, mass, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.orbit = [(x, y)]
        self.velocity_orbit = [(vx, vy)]


def update_position(body, dt):
    body.x += body.vx * dt
    body.y += body.vy * dt


def update_velocity(body, force, dt):
    ax = force[0] / body.mass
    ay = force[1] / body.mass
    body.vx += ax * dt
    body.vy += ay * dt


def gravitational_force(body1, body2, eps=1e-5):
    # Displacements.
    G = 1.0
    dx = body2.x - body1.x
    dy = body2.y - body1.y

    # Distances.
    distance_squared = dx ** 2 + dy ** 2 + eps*eps
    distance = math.sqrt(distance_squared)


    # Forces.
    force_magnitude = G * body1.mass * body2.mass / distance_squared
    force_x = force_magnitude * dx / distance
    force_y = force_magnitude * dy / distance

    return (force_x, force_y)

def compute_force_sum(bodies):
    force_sum = np.zeros((len(bodies), 2))
    iteration = 0
    for body in bodies:
        # Sum forces.
        for other_body in bodies:
            if body != other_body:
                force = gravitational_force(body, other_body)
                force_sum[iteration, 0] += force[0]
                force_sum[iteration, 1] += force[1]
        iteration += 1
    return force_sum

#Run ONE time-step
def simulate(bodies, dt):
    # Find forces
    force_sum = compute_force_sum(bodies)
    #Find new Positions
    for i, body in enumerate(bodies):
        update_velocity(body, force_sum[i, :], dt/2)
        update_position(body, dt)
        body.orbit.append((body.x, body.y))
    # Find updated forces
    force_sum = compute_force_sum(bodies)
    #Find new Velocities
    for i, body in enumerate(bodies):
        update_velocity(body, force_sum[i, :], dt/2)
        body.velocity_orbit.append((body.vx, body.vy))


# Animation code.
def animate(frame, bodies, ax, dt):
    simulate(bodies, dt)
    ax.clear()

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    for body in bodies:
        ax.scatter(
            body.x,
            body.y,
            color=body.color,
            s=100,
            label=f'{body.color} body'
        )
        updated_points = list(zip(*body.orbit))
        ax.plot(updated_points[0], updated_points[1], color=body.color, linewidth=2)


#Check if the system entered a ending condition:
def check_condition(bodies, max_distance_squared, min_distance_squared, eps=1e-5):
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
    if np.max(distances_squared) > max_distance_squared:
        label = 2
        condition = False
    elif np.min(distances_squared) < min_distance_squared:
        label = 0
        condition = False
    else:
        condition = True
        label = 1
    return condition, label

#Run a full simulation
def full_simulation(bodies, dt, Tmax, max_distance_squared, min_distance_squared):
    t = 0
    condition = True
    label = -1
    while t < Tmax and condition:
        simulate(bodies, dt)
        t += dt
        condition, label = check_condition(bodies, max_distance_squared, min_distance_squared)
    return label, t



#Physical Parameters:
dt = 0.01
max_distance_squared = 200
min_distance_squared = 0.001

#Bodies Parameters
Lmax = 1.5
#v = 1
radius = 0.1

#Simulation Parameters:
N = 2400 #Number of Simulations
Tmax = 10 #Maximum iteration of a single simulation
maxSimulation = 1e7 #maximum number of simulations
cutCondition = 50 #minimum number of points in the simulation


#Create the dataset:
def multiple_simulations(N, Tmax, max_distance_squared, min_distance_squared, dt, radius, Lmax, maxSimulation, cutCondition):
    n_steps_max = int(np.ceil(Tmax/dt)) + 2 #adding initial condition to the orbit
    positionsArray = []
    labelArray = []

    simulation = 0 #number of available simulation
    label = 0 #initialize label for balanced dataset

    while simulation < maxSimulation and len(positionsArray) < N:
        simulation += 1 #to stop iterating after too much time
        # Initial random values for the position:
        r = np.random.uniform(0, 2 * np.pi, size=2)
        L = Lmax * np.sqrt(np.random.uniform(0.7, 1, size=2))
        x = L * np.cos(r)
        y = L * np.sin(r)

        body_A = Body(x[0], y[0], radius, "red", 2, 0.0, 0.0)
        body_B = Body(-x[0], y[1], radius, "green", 2, 0.0, 0.0)
        body_C = Body(-(2*x[0] - 2*x[0]), -(2*y[0] + 2*y[1]), radius, 'blue', 1, 0.0, 0.0)

        #Data Sample Simulation
        bodies = [body_A, body_B, body_C]
        new_label, period = full_simulation(bodies, dt, Tmax, max_distance_squared, min_distance_squared)


        n_steps =  len(np.array(bodies[0].orbit)) #adding initial condition to the orbit
        if n_steps > n_steps_max: #Check the orbit length is not too large
            raise ValueError(f"The number of steps ({n_steps}) exceeds maximum allowed ({n_steps_max}).")

        if n_steps < cutCondition: #Check if the orbit length is not too small
            continue
        '''
        if label == new_label: #Uncomment for balanced dataset
            r = random.random()
            if r < 0.085:
                label = 1
            elif r < 0.68:
                label = 0
            else:
                label = 2
        else:
            continue
        '''
        #Plot only for 4 bodies since the CM is conserved
        orbit_matrix = torch.zeros((4, n_steps), dtype=torch.float64)
        for j in range(len(bodies)-1):
            # turn list of (x,y) into an (n_steps,2) array
            orbit = np.array(bodies[j].orbit)
            orbit_matrix[2 * j, :] = torch.from_numpy(orbit[:, 0])
            orbit_matrix[2 * j + 1, :] = torch.from_numpy(orbit[:, 1])
        positionsArray.append(orbit_matrix)
        labelArray.append(new_label)
        #print("number of steps: ", n_steps) #show how long was the orbit if needed
    torch.save({'inputs': positionsArray, 'labels': labelArray}, 'trainInput3BodySIMPLE_try.pt') #Save new inputs in a file



multiple_simulations(N, Tmax, max_distance_squared, min_distance_squared, dt, radius, Lmax, maxSimulation, cutCondition)



'''
data = torch.load('trainInput3BodySIMPLE.pt') #Recover the saved data

print(len(data["inputs"]))
'''
