from setuptools import setup

setup(
    name='robust_face_landmarks_detector',
    version='1.0',
    packages=['face_marks_2d3d'],
    url='',
    license='',
    author='andrey rybin',
    author_email='binrey@icloud.com',
    description='face landmarks detector for 2D and 3D',
    install_requires=["tensorflow==2.2.0", "opencv-python", "numpy"]
)
