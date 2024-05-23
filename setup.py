from setuptools import find_packages
from setuptools import setup

package_name = 'spot_audio'
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name])],
    py_modules=[],
    install_requires=['setuptools'],
    author='Nick Walker',
    author_email='nswalker@cs.washington.edu',
    keywords=['ROS'],
    description='ROS interface for respeaker microphone, generic speaker, and text-to-speech.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'respeaker_node = spot_audio.respeaker_node:main',
            'speaker_node = spot_audio.speaker_node:main',
        ],
    },
)
