# nerfies-to-3d-generative


## Install

First install poetry: https://python-poetry.org/docs/#installation.

Then run

```
poetry shell
poetry install
pip install -r requirements.txt --find-links "https://download.pytorch.org/whl/torch_stable.html https://storage.googleapis.com/jax-releases/jax_releases.html"
```


## Usage

Open the virtualenv shell.

```
poetry shell
```


To extract a rendered Nerfie and point cloud, first download the preprocessed video to `<data-dir>`.
Then

```
python src/neural_pseudo_scan/sample_nerfie.py sample-points --data-dir <data-dir> --output-dir <output-dir> --point-cloud-filename <point-cloud-filename> --train-dir <train-dir>
```


E.g.,

```
python src/neural_pseudo_scan/sample_nerfie.py sample-points --data-dir ./data/capture2/ --output-dir mustafa-rendered-nerfie --point-cloud-filename mustafa.npz --train-dir ./data/capture2/exp2
```


To then visualize the point cloud run

```
python src/neural_pseudo_scan/sample_nerfie.py visualize-point-cloud --point-cloud-path <point-cloud-path>
```


E.g.,

```
python src/neural_pseudo_scan/sample_nerfie.py visualize-point-cloud --point-cloud-path ./mustafa-rendered-nerfie/mustafa.npz
```
