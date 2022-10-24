# Classifier of Cake Types
-----------------------------------

Using pretrained models (VGG, ResNet, MobileNet, InceptionNet, EfficientNet) to classify photos of 5 different types of cakes:
Chocolate, Red Velvet, Apple Pie, French Toast and Garlic Bread.
## Installation

From the source directory run the following commands

### Virtual Env Creation & Activation

- `python3 -m venv venv` for initialising the virtual environment
- `source venv/bin/activate` for activating the virtual environment

### Dependency Installation

The following commands shall be ran **after activating the virtual environment**.

- `pip install --upgrade pip` for upgrading the pip
- `pip install -r requirements.txt` for the functional dependencies
- `pip install -r requirements-dev.txt` for the development dependencies. (should include `pre-commit` module)
- `pre-commit install` for installing the precommit hook

For the extra modules, which are not a standard pip modules (either from your own `src` or from any github repo)

- `pip install -e .` for the files/modules in `src` to be accessed as a package. This is accompanied with `setup.py` and `setup.cfg` files
  - `-e` means installing a project in _editable_ mode, thus any local modifications made to the code will take effect without reinstallation.
