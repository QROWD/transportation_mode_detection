from setuptools import setup

setup(
    name='transportation_mode_detection',
    version='0.0.1',
    packages=[''],
    url='https://github.com/QROWD/transportation_mode_detection',
    license='GNU General Public License v3.0',
    author='Patrick Westphal',
    author_email='patrick.westphal@informatik.uni-leipzig.de',
    description='',
    install_requires=[
        'scikit-learn==0.20.2',
        'pandas==0.23.4',
        'numpy==1.15.4',
        'matplotlib==3.0.0',
        'more-itertools==5.0.0',
        'torch==1.0.0',
        'torchvision==0.2.1',
        'xgboost==0.81',
        'statsmodels==0.9.0',
        'scipy==1.10.0'
    ]
)
