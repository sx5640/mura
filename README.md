# Replicate MURA baseline and playing around

Mura competition https://stanfordmlgroup.github.io/competitions/mura/

#### Please Use Pull-Request If You Want To Update Master Branch, And Add sx5640 As Reviewer
## How to run the script
1. Setup your preferred environment, i.e. virtualenv, conda, docker
2. Install packages: `pip install -r requirements.txt`
3. Download MURA dataset into `/dataset/` directory. Link to MURA: https://stanfordmlgroup.github.io/competitions/mura/
5. To run a model on MURA dataset, run `python {path-to-model-python-file} train`. Add `-h` flag for all options.
7. To show a visualization of model attention or activation of a layer, run `python visualize.py`. Add `-h` flag for all options.

#### Note: 
- In order to run tensorflow in cpu, install `tensorflow` instead of `tensorflow-gpu` in `requirements.txt`
- For MacOS user, importing `dataset` might throw exception, as explained by: https://matplotlib.org/faq/osx_framework.html
  If you use `virtualenv`, one solution pointed out by Kevin is to add the following code to the top of your `dataset.py`:
  ```python
  import matplotlib as mpl
  mpl.use('TkAgg')
  ```
 - The main script does not work with `keras-2.2.0` due to a known bug in keras, but will work with `keras-2.2.2`. 