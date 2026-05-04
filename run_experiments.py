import subprocess
import sys
import os

def run_command(command):
    print(f"\n{'='*45}")
    print(f"Executing: {' '.join(command)}")
    print(f"{'='*45}\n")
    
    try:
        # Use sys.executable to ensure we use the same python environment
        full_command = [sys.executable] + command
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

def main():
    # Define experiments as a list of argument lists
    experiments = [
        ["main.py", "--model_type", "hybrid", "--ordering", "atomic_number", "--epochs", "45"],
        ["main.py", "--model_type", "gin", "--epochs", "45"],
        ["main.py", "--model_type", "hybrid", "--ordering", "electronegativity", "--epochs", "45"],
        ["main.py", "--model_type", "hybrid", "--ordering", "canonical", "--epochs", "45"],
        ["main.py", "--model_type", "hybrid", "--ordering", "learned", "--epochs", "45"],
        ["main.py", "--model_type", "hybrid", "--ordering", "degree", "--epochs", "45"],
    ]

    print(f"Starting {len(experiments)} Tox21 Mamba experiments...")

    for i, cmd_args in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}")
        run_command(cmd_args)

    print(f"\n{'='*45}")
    print("All experiments completed successfully!")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()
