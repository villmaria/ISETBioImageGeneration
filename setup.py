from setuptools import setup

setup(
    name='ISETBioDatasetGeneration',
    version='0.1',
    packages=["ERICA",
               "ISETBio_Retinas", 
               "MATLAB", 
               "PostImageAberration",  "PostImageAberration.Testing", 
               "AberrationRendering"],
    package_data={'':['*']},
    include_package_data=True
)