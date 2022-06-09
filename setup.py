from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager',
      version='1.0.4',
      description='PyTorch training manager (v1.0.4)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=[
            'torchmanager',
            'torchmanager_core',
            'torchmanager_core.devices',
            'torchmanager_core.view',
            'torchmanager.callbacks',
            'torchmanager.losses',
            'torchmanager.metrics',
            'torchmanager.train',
      ],
      package_dir={
            'torchmanager': 'lib',
            'torchmanager_core': 'core',
            'torchmanager_core.devices': 'core/devices',
            'torchmanager_core.view': 'core/view',
            'torchmanager.callbacks': 'lib/callbacks',
            'torchmanager.losses': 'lib/losses',
            'torchmanager.metrics': 'lib/metrics',
            'torchmanager.train': 'lib/train',
      },
      install_requires=[
            'torch>=1.8.2',
            'tqdm',
      ],
      python_requires=">=3.8",
      url="https://github.com/kisonho/torchmanager.git"
)
