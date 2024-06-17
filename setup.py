from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file:str)->List[str]:
    '''will return list of requirements for this package'''
    
    requirements = []
    with open(file) as f_obj:
        requirements = f_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(
    name='ML_deployment',
    version='0.0.1',
    author='Anish Hilary',
    author_email='anishhilary97@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
