# scale_morphology
Shape analysis of Zebrafish scales.

## Motivation
Fish scales are bones - the shape of a fish's scales tells us about bone growth in the organism.
We can use this to measure the effect of mutations, age, sex, etc. on bone growth and regeneration.

## Quick Start
1. Install `uv`.
2. Run `uv sync` to install the environment.
3. Mount the RDSF to get access to the data.
4. Edit the config file `config.yml` to point at your RDSF mount.
5. Run the notebook: `src/notebooks/carran_analysis/1-shape_analysis.ipynb`.

<details>
<summary>New to uv?</summary>
uv is a python package manager that makes setup easy -
like conda, it manages virtual environments, but is much faster
and easier to use.

> Download `uv` from [here](https://docs.astral.sh/uv/#installation).
</details>

<details>
<summary>Mounting the RDSF</summary>
The RDSF (Research Data Storage Facility) is Bristol Uni's research
data repository.

You will need to be added to the Zebrafish_Osteoarthritis project
in order to read data from it - ask your PI for access if you
don't have this already.

There are [several ways](https://www.bristol.ac.uk/acrc/research-data-storage-facility/accessing-the-rdsf/)
for users to access the RDSF; if you're on Linux I recommend:

`gio mount smb://rdsfcifs.acrc.bris.ac.uk/Zebrafish_Osteoarthritis`

It will then prompt you for:
- username (use your UoB username)
- workgroup (`UOB`)
- password ()

If you're on Windows, I recommend using installing WSL
(Windows Subsystem for Linux), cloning this project there and
using the Linux CLI for this project.
</details>
