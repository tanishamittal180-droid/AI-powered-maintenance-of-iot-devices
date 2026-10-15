# Project Notes

## Intended demonstration
This project demonstrates an end-to-end ML workflow: data generation, cleaning, training, evaluation, inference, monitoring UI, and REST API integration.

## Why synthetic data?
The author has no access to industrial hardware. Synthetic telemetry provides a repeatable simulation that can be run on a laptop. It is not a substitute for validated industrial data.

## Model assumptions
- Each row represents one sensor snapshot.
- The label is generated from a synthetic risk formula.
- Random Forest is chosen for approachable tabular classification and feature importance.
- Median imputation is inside the model pipeline so inference uses the same preprocessing.

## Safety
This is an educational proof-of-work project. Never connect the demo prediction directly to industrial controls. Use independent safety interlocks and qualified engineering review for real machinery.
