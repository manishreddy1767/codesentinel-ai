"""
Test script for trainer utilities.
"""

import torch

from backend.ml.training.trainer import aggregate_chunk_logits


print("=" * 60)
print("TRAINER TEST")
print("=" * 60)


# Simulate chunk-level logits for 3 functions.
#
# Function 1 -> 2 chunks
# Function 2 -> 3 chunks
# Function 3 -> 1 chunk

chunk_logits = torch.tensor([
    0.2,
    0.4,

    0.8,
    1.0,
    1.2,

    -0.5,
])

chunk_counts = [2, 3, 1]


print("\nChunk logits:")
print(chunk_logits.tolist())

print("\nChunk counts:")
print(chunk_counts)


function_logits = aggregate_chunk_logits(
    chunk_logits,
    chunk_counts,
)


print("\nFunction-level logits:")
print(function_logits.tolist())


expected_logits = torch.tensor([
    0.3,
    1.0,
    -0.5,
])


assert torch.allclose(
    function_logits,
    expected_logits,
)


print("\nExpected logits:")
print(expected_logits.tolist())


print("\n" + "=" * 60)
print("CHUNK AGGREGATION TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
