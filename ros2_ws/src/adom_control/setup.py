from glob import glob
from setuptools import find_packages, setup

package_name = "adom_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ADOM Team",
    maintainer_email="adom@example.com",
    description="Ackermann keyboard teleop and PCA9685 PWM control for ADOM.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pca9685_control = adom_control.pca9685_control:main",
            "keyboard_teleop = adom_control.keyboard_teleop:main",
            "gamepad_control = adom_control.gamepad_control:main",
        ]
    },
)
