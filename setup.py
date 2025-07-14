from setuptools import setup, file_packages

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]

setup(
    name='Job Success Prediction',
    version='0.0.1',
    author='Niharika',
    author_email='niharikapoduval29@gmail.com',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'pandas',
    ]
)