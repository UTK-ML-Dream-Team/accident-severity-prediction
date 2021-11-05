from setuptools import setup, find_packages, Command
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


# Load Requirements
with open('requirements.txt') as f:
    requirements = f.readlines()

# Load README
with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = []
data_files = ['project_libs/configuration/yml_schema.json']
COMMANDS = []
setup(
    author="drkostas, jheiba, Russtyhub, schoward2, isanjeevsingh",
    author_email="kgeorgio.vols.utk.edu, jlord1@vols.utk.edu, rlimber@vols.utk.edu, "
                 "showar37@vols.utk.edu, ssingh42@vols.utk.edu",
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    data_files=[('', data_files)],
    description="Accident Severity Prediction.",
    entry_points={'console_scripts': COMMANDS},
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='us, accidents, severity, prediction',
    name='accident-severity-prediction',
    packages=find_packages(include=['project_libs',
                                    'project_libs.*']),
    setup_requires=setup_requirements,
    url='https://github.com/UTK-ML-Dream-Team/accident-severity-prediction',
    version='0.1.0',
    zip_safe=False,
)
