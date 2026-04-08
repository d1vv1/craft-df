import numpy as np

data = "/Users/divyanshumahi/Projects/craft-df/processed_dataset/processed_data/fake/0A0IAK9X2W/frame_000000_face_00_dwt.npy"

freq = np.load(data)
print(freq.shape)