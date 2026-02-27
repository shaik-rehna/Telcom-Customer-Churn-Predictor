from setuptools import find_packages, setup


HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->list[str]:

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        # requirements = [req.strip() for req in file_obj]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
name='mlproject',
version='0.0.1',
author='rehnaafroz',
author_email='rehnaafroz@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)