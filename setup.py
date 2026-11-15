# this setup.py will be responsible for creating my machine learning application as a package
# and we can install this package for our project and use it 
from setuptools import find_packages,setup

HYPEN_E_DOT='-e .' #this is because i don't want this to come in this file while run time from requirements.txt

def get_requirements(file_path:str)->list[str]:
    ''' 
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements






setup(
    name='MLProject',
    version='0.0.1',
    author='Ashish Singh',
    author_email='ashishsingh.dsuk@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') ##this is a function which will be used to install any number of packages required for a project
)
