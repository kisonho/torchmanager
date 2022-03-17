from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager',
      version='1.0.2',
      description='PyTorch training manager (v1.0.2)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=[
            'torchmanager',
            'torchmanager.callbacks',
            'torchmanager.core',
            'torchmanager.train',
            'torchmanager_nightly',
            'torchmanager_nightly.train'
      ],
      package_dir={
            'torchmanager': 'lib',
            'torchmanager.callbacks': 'lib/callbacks',
            'torchmanager.core': 'lib/core',
            'torchmanager.train': 'lib/train',
            'torchmanager_nightly': 'nightly',
            'torchmanager_nightly.train': 'nightly/train'
      },
      install_requires=[
            'tensorboard',
            'torch>=1.8.2',
            'tqdm',
      ],
      python_requires=">=3.8",
      url="https://github.com/kisonho/torchmanager.git"
)
