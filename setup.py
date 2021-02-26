from setuptools import setup

with open("requirements.in") as f:
    install_requires = f.read().split()

setup(
    name="bcfind",
    version=2.1,
    description="BCFind is a tool for brain cells localization"
    "from large volumetric images",
    author="Curzio Checcucci, Toresano La Ode, Paolo Frasconi",
    author_email="curzio.checcucci@unifi.it",
    url="https://github.com/lens-biophotonics/BCFind2.1",
    # What does your project relate to?
    keywords="cnn, dog, cell localization, microscopy",
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        "console_scripts": [
            "bcfind-make-data = bcfind.make_training_data:main",
            "bcfind-train = bcfind.train:main",
            "bcfind-evaluate = bcfind.evaluate_train:main",
        ],
    },
)
