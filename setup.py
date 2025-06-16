import subprocess
import sys

import paddle
from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

from setup_ops import get_version

__version__ = get_version()

install_requires = [
    "scipy",
]

test_requires = [
    "pytest",
    "pytest-cov",
]


class CustomCommand:
    """Common functionality for install and develop commands"""

    def run_setup_ops(self):
        try:
            subprocess.check_call([sys.executable, "setup_ops.py", "install"])
        except subprocess.CalledProcessError as e:
            print(f"Error running setup_ops.py: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


class CustomInstallCommand(install, CustomCommand):
    def run(self):
        self.run_setup_ops()
        install.run(self)


class CustomDevelopCommand(develop, CustomCommand):
    def run(self):
        self.run_setup_ops()
        develop.run(self)


include_package_data = True
if paddle.device.cuda.device_count() > 0:
    include_package_data = False

setup(
    name="paddle_sparse",
    version=__version__,
    description=(
        "Paddle Extension Library of Optimized Autograd Sparse "
        "Matrix Operations. Originally from https://github.com/rusty1s/pytorch_sparse."
    ),
    author="Ruibin Cheung",
    author_email="beinggod@foxmail.com",
    keywords=[
        "paddlepaddle",
        "sparse",
        "sparse-matrices",
        "autograd",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
    },
    packages=find_packages(),
    include_package_data=include_package_data,
)
