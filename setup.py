from setuptools import setup, find_packages

requirements_txt = open("requirements.txt").read().split("\n")
requirements = list(filter(lambda x: "--extra" not in x and x is not "", requirements_txt))

setup(
    name="neural_chess",
    version="0.1.0",
    author="Angus Turner",
    author_email="angusturner27@gmail.com",
    url="https://github.com/angusturner/neural_chess",
    description="Chess bot written with Jax and Haiku",
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
)
