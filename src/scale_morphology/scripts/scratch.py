import pathlib
from scale_morphology.scales import read

parent_dir = pathlib.Path("~/zebrafish_rdsf/Rabia/SOST scales").expanduser()
assert parent_dir.exists()

scale_dirs = tuple(d for d in parent_dir.glob("*") if not d.stem in {".DS_Store", "TIFs"})

scale_dir = scale_dirs[1]
scale_paths = scale_dir.glob("*.lif")
path = str(next(scale_paths))

stuff = read.read_lif(path)

name, img = stuff[0]

import matplotlib.pyplot as plt
plt.imshow(img)
plt.title(name)
plt.savefig("scratch.png")

