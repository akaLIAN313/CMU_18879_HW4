import os
from datetime import datetime

def log_experiment_results(X_data, Y_data, script_name, duration=None, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    with open(log_file, "w") as f:
        f.write(f"Script: {script_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        if duration is not None:
            f.write(f"Total Runtime: {duration:.2f} sec ({duration/60:.2f} min)\n")
        f.write("="*50 + "\n")
        f.write("X\tY (Pyr, PV)\tPower\n")
        for x, y in zip(X_data, Y_data):
            f.write(f"{x.tolist()}\t{y.tolist()}\n")
    print(f"Results logged to {log_file}")

