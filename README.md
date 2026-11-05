# Neural-Corrected Operator Learning for Homogenization and Inverse Design

## Run

```bash
python -m pip install -r requirements.txt
python demo.py
```

To run on a GPU:

```bash
python demo.py --device cuda
```

Configure training with `--size`, `--samples`, `--steps`, and `--seed`.
The trained model is saved to `runs/nco.pt`.
