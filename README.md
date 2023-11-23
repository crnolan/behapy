# behapy
Behavioural neuroscience data wrangling and analysis tools

## Installation

Execute the following from the project folder in a shell with conda on the path:

```bash
conda env create -f environment.yaml
conda activate behapy
pip install -e .
```

## Examples

There is a MedPC event reading example in the examples subfolder. This example by default assumes MedPC data files are in the same folder as the notebook file.


## Preprocessing

Convert TDT source data to BIDS-like raw data format:

`tdt2bids [session_fn] [experiment_fn] [bidsroot]`

Open the preprocessing dashboard and confirm rejected regions of the recording:

`ppd [bidsroot]`

Write the preprocessed data to the `derivatives/preprocess` tree:

`preprocess [bidsroot]`
