import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="midplot",
    version="0.0.2",
    description="Plotting margingales",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/midplot",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=["midplot"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["numpy","pytest"],
    entry_points={
        "console_scripts": [
            "midplot=midplot.__main__:main",
        ]
    },
)
