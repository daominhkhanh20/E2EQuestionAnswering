from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from e2eqavn import __version__


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download('punkt')


with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='e2eqavn',
    cmdclass={'install': Install},
    version=__version__,
    description='e2eqavn is end to end pipeline for question answering',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['e2eqavn'],
    install_requires=required_packages,
    python_requires='>3.6.0',
    author='khanhdm',
    author_email='khanhc1k36@gmail.com',
    entry_points={
        'console_scripts': [
            'e2eqavn = e2eqavn.cli:entry_point'
        ]
    },
)
