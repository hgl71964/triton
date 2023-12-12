import subprocess


def get_gpu_name():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            universal_newlines=True)
        gpu_name = output.strip()
        return gpu_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error: {e}")
