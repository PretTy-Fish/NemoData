# NemoData

A Python module with utility functions to parse and manipulate data generated with [NEMO 3D](https://nanohub.org/resources/12593). Specifically, it can directly read the binary outputs (`.nd_evec_*`, `.nd_wf_*`, `.nd_rAtom_*`, etc.), hence saving the time of converting binary to ASCII and reading ASCII into memory (which is binary).

This is written in 2023 and used throughout my Master of Science (Physics) project at the University of Melbourne.

## Usage

The file `nemodata.py` can be put under the same directory with the code that uses it, or in the `site-packages` folder of the active Python distribution. It can then be imported like any other Python module.

```python
import nemodata as nd
```

I personally use the alias `nd` for this module.

## Dependency

The module depends on `numpy`.

The examples use `matplotlib` additionally to generate plots.
