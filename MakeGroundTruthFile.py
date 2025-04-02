
import os
import glob
import pandas as pd 
import numpy as np 

DIRNAME = os.path.split(__file__)[0]
HOLYGRAIL = '/media/maria/MariaData/DatasetPostImageAberration/DoublePass_Mean0.03_STD0.02'

SAVE = '/home/maria/Desktop/DatasetGeneratorERICA/GroundTruthConcatenator/files'
SAVE_CSV = os.path.join(SAVE, 'DoublePass_Mean0.03_STD0.02.csv')
SAVE_NO_DUPLICATES = os.path.join(SAVE, 'DoublePass_Mean0.03_STD0.02_NoDuplicates.csv')



def FindMetaDataCSV(ConeMosaic, Session):
    Meta = os.path.join(HOLYGRAIL, f'ConeMosaic{ConeMosaic}', f'Session{Session}')
    MetaCSV = os.path.join(Meta, 'Meta_data.csv')
    print(MetaCSV)


    MD = None
    try:
        MD = pd.read_csv(MetaCSV,delimiter=',')
        tmp = MD.loc[:, 'Direction (deg from positive x-axis)':'Max. speed (armin / second)']
        rows = MD.shape[0]
        print(rows)
        for i in range(rows):
            # find the name we are working with 
            Im_DF = 'Image_%03i.bmp'%i
            # make a new name 
            Im_Ext = 'Image_%03i'%i
            new_fname = f'/media/maria/MariaData/DatasetPostImageAberration/DoublePass_Mean0.03_STD0.02/ConeMosaic{ConeMosaic}/Session{Session}/Images/{Im_Ext}.bmp'
            # new_fname = f'ConeMosaic{ConeMosaic}_Session{Session}_{Im_Ext}'
            MD.loc[MD['File name'] == Im_DF, 'File name'] = new_fname
        # If you want to save t he temp file 
    # MD.to_csv('/home/maria/Desktop/MicrosaccadeDetectorERICA/DataLoader/test.csv', index=False)
    except pd.errors.EmptyDataError:
        print(f'That shit stink! {MetaCSV, ConeMosaic, Session}')

    
    return MD




    
def main():

    Cones = np.linspace(0, 5, 6, dtype=int)
    Sesh = np.linspace(0, 1, 2, dtype=int)
    print(Cones, '\n',Sesh)
    print(len(Cones), len(Sesh))

    # HolyGrailMetaData = pd.DataFrame()
    for i in range(len(Cones)):
        for j in range(len(Sesh)):
            
            CurrMD = FindMetaDataCSV(i, j)

            if i == 0 and j == 0:
                HolyGrailMetaData = CurrMD
                continue
            else:
                HolyGrailMetaData = pd.concat([HolyGrailMetaData, CurrMD], ignore_index=True)
            # HolyGrailMetaData.append(CurrMD)

    HolyGrailMetaData.to_csv(SAVE_CSV, index=False)

    GroundTruthNoDuplicates = HolyGrailMetaData.drop_duplicates(subset='File name', keep='last')
    GroundTruthNoDuplicates.to_csv(SAVE_NO_DUPLICATES, index=False)

    return 0 

if __name__ == '__main__':
    main()