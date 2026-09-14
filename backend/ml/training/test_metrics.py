"""
Test script for evaluation metrics.
"""

from backend.ml.training.metrics import calculate_metrics


targets = [1, 0, 1, 0, 1, 0]
probabilities = [0.9, 0.1, 0.8, 0.3, 0.6, 0.7]

results = calculate_metrics(targets, probabilities)

print("=" * 60)
print("METRICS TEST")
print("=" * 60)

for name, value in results.items():
    print(f"{name}: {value}")

print("\nMetrics test completed successfully.")
