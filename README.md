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

behapy is maintained by [Chris Nolan](https://github.com/crnolan).

This project would not exist without the energetic contributions from folks in the BrainHack community. Thanks in particular to:

* [Thomas Burton](https://github.com/thomasjburton)
* [Karly Turner](https://github.com/karlyturner)
* [Phil JRDB](https://github.com/philjrdb)
* [Dylan Black](https://github.com/dylanablack)
* [Ilya Kuzovkin](https://github.com/kuz)
* [Daniel Naoumenko](https://github.com/dnao)
* [Almudena Ramirez](https://github.com/almudena607)
* [Joanne Gladding](https://github.com/jmgladding)
* [Lydia Barnes](https://github.com/lydiabarnes01)
* J Bertran-Gonzalez
