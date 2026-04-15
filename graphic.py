import matplotlib.pyplot as plt

# Extracted from your MLX terminal logs
iterations = [1, 200, 400, 600, 800, 1000]
train_loss = [2.094, 0.730, 0.378, 0.222, 0.124, 0.087] # Iter 1 train loss approximated from Iter 10
val_loss = [3.722, 1.243, 1.545, 1.837, 2.053, 2.122]

plt.figure(figsize=(10, 6))
plt.plot(iterations, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(iterations, val_loss, label='Validation Loss', color='red', marker='o')

plt.axvline(x=200, color='green', linestyle='--', label='Optimal Checkpoint (Iter 200)')

plt.title('Qwen 0.5B LoRA Fine-Tuning: Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_curve.png')
print("Chart saved as loss_curve.png!")