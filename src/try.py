from pathlib import Path
import os
files = os.listdir('./data')

for file in files:
    print(Path(file).stem)