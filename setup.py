from setuptools import setup
import os
from glob import glob

package_name = 'rx26_roa'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),       # marker file
        ('share/' + package_name, ['package.xml']),  # package.xml install
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    entry_points={
        'console_scripts': [
            'apf_node = rx26_roa.apf_node_v3:main',
        ],
    },
)
