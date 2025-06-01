import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob
import uuid
from tqdm import tqdm


def generate_realizations(base, num=10, noise_level=0.5):
    base = base.copy()
    realizations = [base]
    for _ in range(1, num):
        noise = np.random.randn(len(base)) * noise_level
        perturb = np.convolve(noise, np.ones(20)/20, mode='same')  # Smooth noise
        perturbed = base + perturb
        realizations.append(perturbed)
    return np.array(realizations)

def process_raw_file(path, window_size=600, min_valid=610, max_chunks=20):
    df = pd.read_csv(path)
    x_raw = df['VS_APPROX_adjusted'].values
    z_raw = df['HORIZON_Z_adjusted'].values

    # Interpolate on 1-foot grid
    x_new = np.arange(0, x_raw.max(), 1.0)
    f_interp = interp1d(x_raw, z_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
    z_new = f_interp(x_new)

    rows = []
    if len(z_new) < min_valid:
        return rows

    num_chunks = 0
    attempts = 0
    while num_chunks < max_chunks:
        start = np.random.randint(0, len(z_new) - window_size)
        chunk = z_new[start:start + window_size].copy()
        chunk -= chunk[299]  # normalize so Z(0)=0

        # Simulate drilling by hiding part of left context
        if np.random.rand() > 0.6:
            hide_up_to = np.random.randint(0, 250)
        else:
            hide_up_to = 0
        chunk_with_nans = chunk.copy()
        chunk_with_nans[:hide_up_to] = np.nan

        realizations = generate_realizations(chunk[300:], num=1) # 10
        output = {
            'geology_id': str(uuid.uuid4()),
        }
        
        for i in range(300):
            output[str(i - 299)] = chunk_with_nans[i]
        for i in range(300):
            output[str(i + 1)] = realizations[0, i]
        
        rows.append(output)
        num_chunks += 1
        attempts += 1

    return rows
