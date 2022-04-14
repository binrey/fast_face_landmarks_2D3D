from setuptools import setup

requirements = [
    "opencv-python",
    "matplotlib",
    "opencv-python",
    "imgaug",
    "face_alignment"
]

setup(
    name='fast_face_landmarks_2D3D',
    version='1.0',
    packages=[],
    url='',
    license='',
    author='andrey rybin',
    author_email='binrey@icloud.com',
    description='face landmarks detector for 2D and 3D',
    install_requires=requirements
)
