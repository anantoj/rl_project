from setuptools import find_packages, setup
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

install_requires =[
    "torch",
    "gym==0.25.2",
    "gym[classic_control]",
    "torchvision"
]

if __name__ == "__main__":
    setup(
        name="rl_project",
        description="Reinforcement Learning Project DQN Benchmark Trainer",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="anantoj",
        author_email="gg.ananto@gmail.com",
        url="https://github.com/anantoj/rl-project",
        license="Apache License",
        packages=find_packages(),
        install_requires=install_requires,
        include_package_data=True,
        platforms=["linux", "unix", "windows"],
        python_requires=">=3.6",
    )