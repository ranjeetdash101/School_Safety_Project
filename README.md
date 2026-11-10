# School Safety Emergency Alert Project

This is a separate standalone Python project for classifying school emergency severity levels.

## Files

- `school_safety_project.py` – main project script
- `SchoolSafetyEmergencyDataset.csv` – dataset
- `requirements.txt` – Python packages

## Run

```bash
pip install -r requirements.txt
python school_safety_project.py train
python school_safety_project.py predict --temperature 40 --smoke 70 --co 25 --gas 78 --sound 95 --crowd 50 --motion 60 --flame 1 --duration 7 --incident-type "Heavy Smoke"
```

## Goal

Predict whether a school event is safe, a normal alert, or a high-priority emergency.
