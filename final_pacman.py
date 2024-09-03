from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import tkinter as tk
import numpy as np
from numpy import pi
import random
from queue import PriorityQueue

# Define the maze
maze = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# Initialize quantum circuit with 21 qubits (based on the number of 0s) + 3 additional qubits for Pacman and ghosts
num_qubits = 21
qc = QuantumCircuit(num_qubits, num_qubits)

# Map the maze and initialize qubits
qubit_index = 0
qubit_mapping = {}
food_qubits = set()

for row in range(len(maze)):
    for col in range(len(maze[row])):
        if maze[row][col] == 0:
            qubit_mapping[(row, col)] = qubit_index
            qubit_index += 1

# Assign Pacman and ghosts to qubits
pacman_qubit = 18
ghost1_qubit = 19
ghost2_qubit = 20

# Place Pacman and ghosts on the maze
pacman_position = (1, 1)
ghost1_position = (1, 5)
ghost2_position = (5, 1)

# Initialize their positions to |1⟩ state
qc.x(qubit_mapping[pacman_position])

# Initialize Pacman to |1⟩ state (default state)
qc.x(pacman_qubit)

# Initialize Pacman and ghosts qubits to |0⟩ state (default state)
# Entangle the two ghosts
qc.h(ghost1_qubit)
qc.cx(ghost1_qubit, ghost2_qubit)
qc.reset(ghost1_qubit)
qc.reset(ghost2_qubit)

# Measure all qubits
qc.measure(range(num_qubits), range(num_qubits))

simulator = Aer.get_backend('qasm_simulator')
tqc = transpile(qc, simulator)
result = simulator.run(tqc, shots=1024).result()

# Get the measurement results
counts = result.get_counts(qc)

# Determine the color of Pacman and ghosts based on measurement results
pacman_color = "yellow"
ghost1_color = "red"
ghost2_color = "red"

# Check the measurement results for each shot
for outcome, count in counts.items():
    pacman_measurement = outcome[-(pacman_qubit + 1)]
    ghost1_measurement = outcome[-(ghost1_qubit + 1)]
    ghost2_measurement = outcome[-(ghost2_qubit + 1)]

    if pacman_measurement == '0':
        pacman_color = "blue"
    if ghost1_measurement == '1':
        ghost1_color = "green"
    if ghost2_measurement == '1':
        ghost2_color = "green"

def get_sv():
    simulation = Aer.get_backend('statevector_simulator')
    tqcs = transpile(qc, simulation)
    result = simulation.run(tqcs).result()

    # Get the statevector
    statevector = result.get_statevector()

    # Extract the amplitude for Pacman's qubit in state |1⟩
    amplitude = statevector[pacman_qubit]

    # Calculate the phase of Pacman's qubit
    phase = np.angle(amplitude)
    degrees = np.degrees(phase)

    return degrees

def get_qubit_index(point):
    return qubit_mapping.get(point, "Point not found")

def a_star_search(start, goal):
    # A* pathfinding algorithm
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {point: float('inf') for point in qubit_mapping.keys()}
    g_score[start] = 0
    f_score = {point: float('inf') for point in qubit_mapping.keys()}
    f_score[start] = heuristic(start, goal)
    
    while not open_set.empty():
        _, current = open_set.get()
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= neighbor[0] < len(maze) and
                    0 <= neighbor[1] < len(maze[0]) and
                    maze[neighbor[0]][neighbor[1]] == 0):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))
    
    return []

def move_ghosts():
    global ghost1_position, ghost2_position, pacman_position
    path_ghost1 = a_star_search(ghost1_position, pacman_position)
    path_ghost2 = a_star_search(ghost2_position, pacman_position)
    
    if len(path_ghost1) > 1:
        next_position_ghost1 = path_ghost1[1]
    else:
        next_position_ghost1 = ghost1_position
    
    if len(path_ghost2) > 1:
        next_position_ghost2 = path_ghost2[1]
    else:
        next_position_ghost2 = ghost2_position
    
    if ghost1_position != next_position_ghost1:
        ghost1_position = next_position_ghost1

    if ghost2_position != next_position_ghost2:
        ghost2_position = next_position_ghost2

def check_game_over():
    global pacman_position, ghost1_position, ghost2_position
    if pacman_position == ghost1_position or pacman_position == ghost2_position:
        win_label.config(text="Game Over! Press 'R' to restart")
        return True
    return False

def draw_maze():
    global pacman_position, ghost1_position, ghost2_position, pacman_color, ghost1_color, ghost2_color

    canvas.delete("all")
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            if (row, col) == pacman_position:
                color = pacman_color
            elif (row, col) == ghost1_position:
                color = ghost1_color
            elif (row, col) == ghost2_position:
                color = ghost2_color
            else:
                color = "black" if maze[row][col] == 1 else "white"

            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

            # Draw a small green square inside white blocks with food
            if qubit_mapping.get((row, col)) in food_qubits:
                food_size = cell_size // 5  # Size of the green square
                fx1 = x1 + (cell_size - food_size) // 2
                fy1 = y1 + (cell_size - food_size) // 2
                fx2 = fx1 + food_size
                fy2 = fy1 + food_size
                canvas.create_rectangle(fx1, fy1, fx2, fy2, fill="green", outline="", tags="food")

    window.update()

def restart_game():
    global pacman_position, ghost1_position, ghost2_position, food_qubits
    pacman_position = (1, 1)
    ghost1_position = (1, 5)
    ghost2_position = (5, 1)
    food_qubits = place_food()
    draw_maze()

def place_food():
    # Generate list of all valid positions
    valid_positions = [pos for pos in qubit_mapping.keys() if pos != pacman_position and pos != ghost1_position and pos != ghost2_position]
    
    # Randomly select a subset of valid positions for food
    num_food = len(valid_positions) // 2
    food_positions = random.sample(valid_positions, num_food)
    
    # Map food positions to qubits
    return set(qubit_mapping[pos] for pos in food_positions)

def on_key_press(event):
    global pacman_position, food_qubits, win_label
    degs = get_sv()

    if event.keysym == 'Up':
        pacman_position_new = (pacman_position[0]-1, pacman_position[1])
        if degs == 90.0:
            qc.p(-pi/2, pacman_qubit)
        elif degs == -180.0:
            qc.z(pacman_qubit)
        elif degs == -90.0:
            qc.s(pacman_qubit)

    elif event.keysym == 'Right':
        pacman_position_new = (pacman_position[0], pacman_position[1]+1)
        if degs == 0.0:
            qc.s(pacman_qubit)
        elif degs == -180.0:
            qc.p(-pi/2, pacman_qubit)
        elif degs == -90.0:
            qc.z(pacman_qubit)

    elif event.keysym == 'Down':
        pacman_position_new = (pacman_position[0]+1, pacman_position[1])
        if degs == 0.0:
            qc.z(pacman_qubit)
        elif degs == 90.0:
            qc.s(pacman_qubit)
        elif degs == -90.0:
            qc.p(-pi/2, pacman_qubit)

    elif event.keysym == 'Left':
        pacman_position_new = (pacman_position[0], pacman_position[1]-1)
        if degs == 0.0:
            qc.p(-pi/2, pacman_qubit)
        elif degs == 90.0:
            qc.z(pacman_qubit)
        elif degs == -180.0:
            qc.s(pacman_qubit)

    elif event.keysym == 'r':
        restart_game()
        return  # Exit the function to avoid further processing in case of restart

    # Check if the new position is valid (inside the maze and not a wall)
    if (0 <= pacman_position_new[0] < len(maze) and
            0 <= pacman_position_new[1] < len(maze[0]) and
            maze[pacman_position_new[0]][pacman_position_new[1]] == 0):

        # Perform the qubit swap
        old_index = get_qubit_index(pacman_position)
        new_index = get_qubit_index(pacman_position_new)
        qc.swap(old_index, new_index)

        # Apply the Y gate to simulate eating food
        if new_index in food_qubits:
            qc.y(new_index)
            food_qubits.remove(new_index)

            if not food_qubits:
                win_label.config(text="You Win! Press 'R' to restart")
                return

        pacman_position = pacman_position_new  # Update Pacman's position
        move_ghosts()

        if check_game_over():  # Check if game over
            return

        draw_maze()

# Initialize GUI
window = tk.Tk()
window.title("Quantum Pacman")

canvas_width = len(maze[0]) * 40
canvas_height = len(maze) * 40
cell_size = 40

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()

win_label = tk.Label(window, text="")
win_label.pack()

window.bind("<KeyPress>", on_key_press)

restart_game()

window.mainloop()