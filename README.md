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

## Contributors

behapy is maintained by [@crnolan].

Open source projects could not exist without the energetic work from all the wonderful community. Thanks in particular to:

* Thomas Burton [@thomasjburton]
* Karly Turner [@karlyturner]
* Phil JRDB [@philjrdb]
* Dylan Black [@dylanablack]
* Ilya Kuzovkin [@kuz]
* Daniel Naoumenko [@dnao]
* Almudena Ramirez [@almudena607]
* Joanne Gladding [@jmgladding]
* Lydia Barnes [@lydiabarnes01]
* J Bertran-Gonzalez
