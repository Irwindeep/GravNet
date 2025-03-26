import numpy as np
from pycbc.types import TimeSeries # type: ignore
from pycbc.waveform import get_td_waveform # type: ignore

def generate_simulated_waveform(
    approximant: str, m1: float, m2: float, f_lower: float,
    delta_t: float
) -> TimeSeries:
    hp, _ = get_td_waveform(
        approximant=approximant, mass1=m1, mass2=m2,
        f_lower=f_lower, delta_t=delta_t, distance=100
    )

    return hp

def adjust_simulated_waveform(waveform: TimeSeries, desired_length: float) -> TimeSeries:
    target_samples = int(desired_length/waveform.delta_t)

    peak_location = np.random.uniform(0.925, 0.975, size=1)
    desired_peak_index = int(peak_location.item() * target_samples)
    
    original_length = len(waveform)
    original_peak_index = np.argmax(np.abs(waveform.data))
    offset = int(desired_peak_index - original_peak_index)
    new_hp = np.zeros(target_samples)

    start_i, end_i = max(0, -offset), min(original_length, target_samples - offset)
    dest_start = max(0, offset)
    dest_end = dest_start + (end_i - start_i)
    new_hp[dest_start:dest_end] = waveform.data[start_i:end_i]
    
    new_hp_ts = TimeSeries(new_hp, delta_t=waveform.delta_t, epoch=0)
    return new_hp_ts
