import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create a simple model with parameters to optimize
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create model and optimizer
model = SimpleModel()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)

# Create the same scheduler as in your Yoga Pose Detection model
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Simulate training for 100 epochs to see multiple cycles
num_epochs = 100
learning_rates = []

# Record learning rates
for epoch in range(num_epochs):
    # Before each epoch, record the learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Step the scheduler
    scheduler.step(epoch)

# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(range(num_epochs), learning_rates)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingWarmRestarts Learning Rate Schedule')

# Mark the restart points
restart_epochs = [0]
current_t0 = 20
while restart_epochs[-1] + current_t0 < num_epochs:
    restart_epochs.append(restart_epochs[-1] + current_t0)
    current_t0 = current_t0 * 2  # T_mult=2 doubles the period each time

for restart in restart_epochs:
    if restart > 0:  # Skip the first restart at epoch 0
        plt.axvline(x=restart, color='r', linestyle='--', alpha=0.5)
        plt.text(restart+1, 0.0004, f'Restart: {restart}', rotation=90)

# Add detailed annotations to explain the behavior
plt.annotate('Initial LR: 0.0005', xy=(2, 0.0005), xytext=(2, 0.00045), 
            arrowprops=dict(facecolor='black', shrink=0.05))
            
plt.annotate('Min LR: 1e-6', xy=(19, 1e-6), xytext=(19, 0.0001), 
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('First cycle: 20 epochs', xy=(10, 0.00025), xytext=(10, 0.0002), 
            arrowprops=dict(facecolor='black', shrink=0.05))
            
if len(restart_epochs) > 1:
    plt.annotate(f'Second cycle: {restart_epochs[1]-restart_epochs[0]} epochs', 
                xy=((restart_epochs[0]+restart_epochs[1])/2, 0.00025), 
                xytext=((restart_epochs[0]+restart_epochs[1])/2, 0.0002), 
                arrowprops=dict(facecolor='black', shrink=0.05))

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig('cosine_annealing_visualization.png')
print("Visualization saved to 'cosine_annealing_visualization.png'")

# Print some specific learning rate values for key epochs to understand the pattern
print("\nLearning rate values at key epochs:")
print(f"Initial learning rate (epoch 0): {learning_rates[0]:.8f}")
print(f"Middle of first cycle (epoch 10): {learning_rates[10]:.8f}")
print(f"End of first cycle (epoch 19): {learning_rates[19]:.8f}")
print(f"Beginning of second cycle (epoch 20): {learning_rates[20]:.8f}")
print(f"Middle of second cycle (epoch 30): {learning_rates[30]:.8f}")
print(f"End of second cycle (epoch 59): {learning_rates[59]:.8f}")
if num_epochs > 60:
    print(f"Beginning of third cycle (epoch 60): {learning_rates[60]:.8f}")

# Calculate when learning rate is at specific thresholds
def find_epochs_at_threshold(lrs, threshold, operation="below"):
    epochs = []
    for i, lr in enumerate(lrs):
        if operation == "below" and lr < threshold:
            epochs.append(i)
        elif operation == "above" and lr > threshold:
            epochs.append(i)
    return epochs

low_lr_epochs = find_epochs_at_threshold(learning_rates, 0.0001, "below")
high_lr_epochs = find_epochs_at_threshold(learning_rates, 0.0004, "above")

print("\nEpochs where learning rate drops below 0.0001:", low_lr_epochs[:10], "...")
print("Epochs where learning rate jumps above 0.0004:", high_lr_epochs[:10], "...")

# Calculate the mathematical formula
print("\nMathematical formula for the learning rate at epoch t:")
print("For the first cycle (epochs 0-19):")
print("  lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(π * t / T_0))")
print("  Where: eta_min = 1e-6, eta_max = 0.0005, T_0 = 20")

print("\nFor the second cycle (epochs 20-59):")
print("  lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(π * (t - T_0) / (T_0 * T_mult)))")
print("  Where: T_0 = 20, T_mult = 2")

print("\nFor the third cycle (epochs 60-139):")
print("  lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(π * (t - T_0 - T_0*T_mult) / (T_0 * T_mult^2)))")
