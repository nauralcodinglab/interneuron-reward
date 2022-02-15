# Reward tuning in motor cortex

<div align="center">
<img
src="https://img.shields.io/badge/python-3.8%20|%203.9-green.svg"
alt="Python >=3.8"/>
<img
src="https://img.shields.io/badge/os-macOS%20|%20linux-green.svg"
alt="macOS or Linux"/>
<a href="https://doi.org/10.7554/eLife.72549">
<img
src="https://img.shields.io/badge/doi-10.7554%2FeLife.72549-informational.svg"
alt="doi: 10.7554/eLife.72549"/>
</a>
</div>

This repository contains code for figure 3 of the following paper:

> Lee C, Harkin EF, Yin X, Naud R, Chen S. Cell-type specific responses to
> associative learning in the primary motor cortex. eLife. 2022;11:e72549.

Open access available [here](https://doi.org/10.7554/eLife.72549).

## Setup

Paste the shell snippet below into a terminal to download the source code,
install required python packages, and set up the required directory structure.
It is recommended to do this in a virtual environment (eg, a new Anaconda
environment) to avoid polluting your python installation.

```sh
mkdir -p interneuron-reward-project/interneuron-reward-data/{processed,raw} && \
    cd interneuron-reward-project && \
    git clone --depth 1 https://github.com/nauralcodinglab/interneuron-reward.git && \
    cd interneuron-reward && \
    pip install -r requirements.txt && \
    pip install -e .
```

Next, download the [raw data](http://dx.doi.org/10.5061/dryad.q573n5tjj) and
place it in the `interneuron-reward-data/raw` directory that was just created.

Set up a database that is compatible with the Python SQLAlchemy package (eg,
SQLite or MySQL) and create a new environment variable called
`SQLALCHEMY_ENGINE_URL` containing a URL that can be used to access it (more
info in the SQLAlchemy
[docs](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls)).
Once this is done, the raw data can be loaded into the database by running the
scripts in `src`.

Finally, launch a Jupyter notebook server and run the Python and R notebooks
under `notebooks` to reproduce results.

## License

If you use this code in a publication, please cite our paper!

```
@article{lee2022cell,
  title={Cell-type specific responses to associative learning in the primary motor cortex},
  author={Lee, Candice and Harkin, Emerson F and Yin, Xuming and Naud, Richard and Chen, Simon X},
  journal={eLife},
  volume={11},
  pages={e72549},
  year={2022},
  publisher={eLife Sciences Publications Limited}
}
```

<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>
<br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
</p>

This software is provided "as-is" in the spirit of the
[CRAPL](https://matt.might.net/articles/crapl/CRAPL-LICENSE.txt)
academic-strength open-source license.
