import subprocess

def get_free_vram():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    # Extract free memory for each GPU
    free_memory = [int(x) for x in result.stdout.strip().split('\n')]
    return free_memory