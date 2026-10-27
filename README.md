# RBM Network Intrusion Detector

Simple web demo for the files in this folder. It replays rows from `traffic_sample.csv`, sends them through the saved RBM in `rbm_artifacts.pkl`, and shows whether each connection looks normal or anomalous.

## Run

```bash
pip install -r requirements.txt
python app.py
```

On Windows you can also run:

```bat
run_app.bat
```

Then open:

```text
http://127.0.0.1:8000
```

## What the app includes

- Live traffic replay with Start, Stop, attack probability, attack type, and attack burst controls.
- Free-energy chart with the trained threshold.
- Alert banner, latest attribution features, and last 10 records.
- Single-record presets loaded from real rows in `traffic_sample.csv`.
- Batch summary metrics for the whole sample file.

## Notes

The app uses the trained artifact as-is. It does not retrain the model, change preprocessing, or generate real network traffic. If scikit-learn prints a version warning when loading the pickle, that is expected because the artifact was created with a newer scikit-learn version.
