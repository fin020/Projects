import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import bernoulli
import matplotlib.animation as animation

# Setup
n = 100
x = np.arange(0, n)
p = 0.5

# Create figure with more space for controls
fig = plt.figure(figsize=(12, 8))

# Create main plot area for random walk
ax1 = plt.axes([0.1, 0.3, 0.85, 0.65])  # [left, bottom, width, height]

# Initialize plot
ax1.set_title(f"Animated Random Walk with p = {p:.3f}")
ax1.set_xlim(0, n)
ax1.set_ylim(-n//2, n//2)
ax1.set_ylabel("Position")
ax1.set_xlabel("Step")
ax1.grid(True, alpha=0.3)

# Create empty line for animation
line, = ax1.plot([], [], 'b-', linewidth=2, label="Random Walk")
current_dot, = ax1.plot([], [], 'ro', markersize=8, label="Current Position")
ax1.legend()

# Add slider for p value
slider_ax = plt.axes([0.15, 0.2, 0.7, 0.03])
p_slider = Slider(
    ax=slider_ax,
    label=r"Probability p",
    valmin=0.0,
    valmax=1.0,
    valstep=0.01,
    valinit=p
)

# Add buttons for control
play_ax = plt.axes([0.15, 0.12, 0.1, 0.04])
pause_ax = plt.axes([0.27, 0.12, 0.1, 0.04])
reset_ax = plt.axes([0.39, 0.12, 0.1, 0.04])

play_button = Button(play_ax, 'Play', color='lightgray')
pause_button = Button(pause_ax, 'Pause', color='lightgray')
reset_button = Button(reset_ax, 'Reset', color='lightgray')

# Add probability distribution plot
ax2 = plt.axes([0.15, 0.02, 0.7, 0.08])
ax2.set_title("Step Probability Distribution")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(0, 1.1)
ax2.set_xticks([-1, 1])
ax2.set_xticklabels(['Down (-1)', 'Up (+1)'])
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.grid(True, alpha=0.3, axis='y')

# Variables for animation
current_step = 0
walk_data = np.zeros(n)
is_animating = False
animation_obj = None

# Function to generate a step based on probability p
def generate_step(current_p):
    if bernoulli.rvs(current_p):
        return 1  # Success: move up
    else:
        return -1  # Failure: move down

# Animation initialization
def init():
    line.set_data([], [])
    current_dot.set_data([], [])
    return line, current_dot

# Animation update function
def animate(frame):
    global current_step, walk_data, is_animating
    
    if not is_animating or current_step >= n:
        return line, current_dot
    
    current_p = p_slider.val
    step = generate_step(current_p)
    
    if current_step == 0:
        walk_data[0] = 0
    else:
        walk_data[current_step] = walk_data[current_step - 1] + step
    
    # Update line plot
    line.set_data(x[:current_step + 1], walk_data[:current_step + 1])
    
    # Update current position dot
    current_dot.set_data([current_step], [walk_data[current_step]])
    
    # Update title
    ax1.set_title(f"Animated Random Walk with p = {current_p:.3f} (Step: {current_step + 1}/{n})")
    
    # Adjust y-limits dynamically
    if current_step > 0:
        current_min = min(walk_data[:current_step + 1])
        current_max = max(walk_data[:current_step + 1])
        margin = max(abs(current_min), abs(current_max)) * 0.2
        ax1.set_ylim(min(current_min, -10) - margin, max(current_max, 10) + margin)
    
    current_step += 1
    
    return line, current_dot

# Control functions
def play_animation(event):
    global is_animating, current_step, animation_obj
    
    if current_step >= n:
        # If we've reached the end, reset first
        reset_animation(None)
    
    is_animating = True
    
    # Create a new animation if one doesn't exist or has finished
    if animation_obj is None:
        animation_obj = animation.FuncAnimation(fig, animate, init_func=init,
                                              frames=n, interval=50, blit=True, repeat=False)

def pause_animation(event):
    global is_animating
    is_animating = False

def reset_animation(event):
    global current_step, walk_data, is_animating, animation_obj
    current_step = 0
    walk_data = np.zeros(n)
    line.set_data([], [])
    current_dot.set_data([], [])
    ax1.set_title(f"Animated Random Walk with p = {p_slider.val:.3f}")
    ax1.set_ylim(-n//2, n//2)
    is_animating = False
    
    # Stop any existing animation
    if animation_obj and animation_obj.event_source:
        animation_obj.event_source.stop()
    animation_obj = None
    
    fig.canvas.draw_idle()

# Create initial probability distribution bars
bar_width = 0.4
bars = ax2.bar([-1, 1], [1-p, p], width=bar_width, 
               color=['red', 'blue'], alpha=0.7)

# Connect controls
play_button.on_clicked(play_animation)
pause_button.on_clicked(pause_animation)
reset_button.on_clicked(reset_animation)

# Update probability distribution when slider changes
def update_probability_dist(val):
    current_p = p_slider.val
    bars[0].set_height(1 - current_p)  # Probability of -1
    bars[1].set_height(current_p)      # Probability of +1
    
    # Update bar colors based on probability
    if current_p > 0.5:
        bars[1].set_color('darkblue')
        bars[0].set_color('darkred')
    else:
        bars[1].set_color('blue')
        bars[0].set_color('red')
    
    # Update title if not animating
    if not is_animating:
        ax1.set_title(f"Animated Random Walk with p = {current_p:.3f}")
    
    fig.canvas.draw_idle()

p_slider.on_changed(update_probability_dist)

plt.show()