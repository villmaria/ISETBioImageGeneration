# ISETBio Retina and ERICA Dataset Generator

This project should encapsulate of the information for generating ERICA datasets using ISETBio retinas.


#### Install the requirements
Navigate to the directory in which you have cloned the repository. 
You will now need to install the requirements of the project/

###### On Unix/Linux:
```
pip install -r requirements.txt
```
###### On Windows
```
python -m pip install -r requirements.txt
```

#### Install the package 

###### On Unix/Linux:
```
pip install -e .
```
###### On Windows
```
python -m pip install -e .
```

#### Run ISETBio simulation 
Check out the matlab code provided in the matlab folder - make sure you download ISETBio and ISETCam. 

#### Run ERICA Simulations 
Handling of compatibility between the results of any ISETBio simulation and ERICA has been describe in the comments of ERICA > RandomAnglesDrift_ISETBioMosaic.py.

One thing to note is that ISETBio always defines matricies in the format (X, Y) whereas ERICA uses (Y, X). 

# Questions?
Please do feel free to email me any questions at maria.villamil@magd.ox.ac.uk

