import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyradox",
    version="0.17.8",
    author="Ritvik Rastogi",
    author_email="rastogiritvik99@gmail.com",
    description="State of the Art Neural Networks for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ritvik19",
    packages=setuptools.find_packages(
        exclude=["ProjectFiles", ".git", ".idea", ".gitattributes", ".gitignore"]
    ),
    install_requires=["Keras>=2.3.1", "numpy>=1.18.1", "tensorflow>=2.2.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)