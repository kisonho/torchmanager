from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='torchmanager-nightly',
    version="1.2b4",
    description="PyTorch Training Manager v1.2 (Beta 4)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kison Ho',
    author_email='unfit-gothic.0q@icloud.com',
    packages=[
        'torchmanager',
        'torchmanager.callbacks',
        'torchmanager.configs',
        'torchmanager.data',
        'torchmanager.losses',
        'torchmanager.metrics',
        'torchmanager.train',
        'torchmanager_core',
        'torchmanager_core.devices',
        'torchmanager_core.errors',
        'torchmanager_core.random',
        'torchmanager_core.view',
    ],
    package_dir={
        'torchmanager': 'torchmanager',
        'torchmanager.callbacks': 'torchmanager/callbacks',
        'torchmanager.configs': 'torchmanager/configs',
        'torchmanager.data': 'torchmanager/data',
        'torchmanager.losses': 'torchmanager/losses',
        'torchmanager.metrics': 'torchmanager/metrics',
        'torchmanager.train': 'torchmanager/train',
        'torchmanager_core': 'torchmanager_core',
        'torchmanager_core.devices': 'torchmanager_core/devices',
        'torchmanager_core.errors': 'torchmanager_core/errors',
        'torchmanager_core.random': 'torchmanager_core/random',
        'torchmanager_core.view': 'torchmanager_core/view',
        'torchmanager_scipy': 'torchmanager_scipy',
        'torchmanager_scipy.metrics': 'torchmanager_scipy/metrics',
        'torchmanager_tensorboard': 'torchmanager_tensorboard',
        'torchmanager_tensorboard.callbacks': 'torchmanager_tensorboard/callbacks'
    },
    install_requires=[
        'torch',
        'tqdm',
    ],
    extra_requires={
        'scipy': ['torchmanager_scipy', 'torchmanager_scipy.metrics'],
        'tensorboard': ['torchmanager_tensorboard', 'torchmanager_tensorboard.callbacks']
    },
    python_requires=">=3.8",
    url="https://github.com/kisonho/torchmanager.git"
)
