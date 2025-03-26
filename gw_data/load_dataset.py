import json
import numpy as np
from gwpy.timeseries import TimeSeries # type: ignore
from gwpy.segments import Segment, SegmentList # type: ignore

from typing import List
from tqdm.auto import tqdm

def interval_intersection(list1: List[List[int]], list2: List[List[int]]) -> List[List[int]]:
    i, j = 0, 0
    intersections = []
    
    while i < len(list1) and j < len(list2):
        a_start, a_end = list1[i]
        b_start, b_end = list2[j]
        
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        
        if start < end: intersections.append([start, end])
        
        if a_end < b_end: i += 1
        else: j += 1
            
    return intersections

with open("l1_o2_segments.json", 'r') as file:
    o2_segments = json.load(file)["segments"]
    no_injection_intval = [
        [1164556817, 1164755435], [1164755459, 1164765305], [1164765357, 1164769204],
        [1164769557, 1178980183], [1178980184, 1187733618]
    ]
    o2_segments = interval_intersection(o2_segments, no_injection_intval)
    
with open("l1_o2_no_cw_inj.json", 'r') as file:
    no_injection_intval = json.load(file)["segments"]
    o2_segments = interval_intersection(o2_segments, no_injection_intval)

duration = sum(segment[1] - segment[0] for segment in o2_segments)

with open("l1_o3b_segments.json", 'r') as file:
    o3b_segments = json.load(file)["segments"]
    
with open("l1_o3b_no_cw_inj.json", 'r') as file:
    no_injection_intval = json.load(file)["segments"]
    o3b_segments = interval_intersection(o3b_segments, no_injection_intval)

duration = sum(segment[1] - segment[0] for segment in o3b_segments)

gw_event_segments = [
    [1264211182, 1264215278], # GW200128_022011
    [1264314069, 1264318165], # GW200129_065458
    [1264622519, 1264626615], # GW200201_203549
    [1264691364, 1264695460]  # GW200202_154313
]

o2_intervals = [Segment(*segment) for segment in o2_segments]
o3b_intervals = [Segment(*segment) for segment in o3b_segments]

observation_intervals = SegmentList(o2_intervals + o3b_intervals)
gw_event_intervals = SegmentList([Segment(*segment) for segment in gw_event_segments])
observation_intervals = observation_intervals - gw_event_intervals

observation_intervals = SegmentList([
    interval for interval in observation_intervals
    if (
        (str(interval[0]).startswith("11") and (interval[1] - interval[0] >= 15000)) or
        (str(interval[0]).startswith("12") and (55000 <= interval[1] - interval[0] <= 70000))
    )
])

duration = sum(interval[1]-interval[0] for interval in observation_intervals)
print(f"Total Observation Duration: {duration:,} sec")

channel_name, errors = "L1", []
for interval in tqdm(observation_intervals, desc="Loading Segments"):
    try:
        data = TimeSeries.fetch_open_data(channel_name, interval[0], interval[1], timeout=300)
        np.save(f"segments/L1-{interval[0]}_{interval[1]}.npy", data.value)
    except Exception as e:
        errors.append(f"Interval: {interval} not Fetched Due to\nError: {e}")

for error in errors:
    print(error)
    print()
