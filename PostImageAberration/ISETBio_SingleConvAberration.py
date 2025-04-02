
import matplotlib.pyplot as plt 
import cv2
from astropy.io import fits
import random
import shutil
import numpy
import yaml
import pandas as pd 
from tqdm import tqdm
import sys
import csv
import os

erica_directory = '/home/maria/Desktop/ERICA'
sys.path.append(erica_directory)
from ERICA import AOSLO_system, fixation_simulation, sampling, generate_retina
from ERICA import ERICA_toolkit, frame_generation, peak_detection
from scipy.special import erf
from PIL import Image

from AberrationRendering import psf_generation_with_aberrations
from scipy.signal import fftconvolve, decimate, convolve2d

save_folder = '/media/maria/MariaData/ISETBioDataset2_0_PostImageAberration_LargerScale'


fov = [1.0, 1.0]
eccentricity_fixation = [2.0, 0.0]

eccentricity_image = [eccentricity_fixation[0]*-1,eccentricity_fixation[1]]


current_directory = os.getcwd()
ground_truth_folder = '/home/maria/Desktop/DatasetGeneratorERICA/ChocolateChip/ISETBio_Retinas/'
# Specify the AOSLO system parameter file to use

# system_parameters_file = os.path.normpath(os.path.join(erica_directory, 'System_parameters', 'system_parameters_realistic.yaml'))
system_parameters_file = os.path.normpath(os.path.join(erica_directory, 'System_parameters', 'system_parameters_demo.yaml'))


cone_reflectance_parameters_file = os.path.normpath(os.path.join(erica_directory, 'Reflectance_parameters', 'cone_reflectance_parameters.yaml'))
sys.stdout.write('Using parameter file: %s\n'%cone_reflectance_parameters_file)
# Open the cone reflectance parameters file
with open(cone_reflectance_parameters_file, 'r') as f:
    cone_reflectance_parameters = yaml.safe_load(f)


save_folder_ground_truth = os.path.normpath(os.path.join(erica_directory, 'Retinal_mosaic_parameters_arrays', system_parameters_file.split('.')[0]))
if os.path.isdir(os.path.normpath(os.path.join(erica_directory, 'Retinal_mosaic_parameters_arrays'))) == 0:
    os.mkdir(os.path.normpath(os.path.join(erica_directory, 'Retinal_mosaic_parameters_arrays')))
if os.path.isdir(save_folder_ground_truth) == 0:
    os.mkdir(save_folder_ground_truth)

sys.stdout.write('Using parameter file: %s'%system_parameters_file)
myAOSLO = AOSLO_system.AOSLO(system_parameters_file)
# Eye movements are generated with a lower temporal resolution and then upsampled at the end. This speeds up 
# data generation. The subsampling number should be small, and no larger than number of pixels per scan line, to avoid 
# undersampling. The default is 64
subsample_rate = 64

# Specify the fixational ye motion parameter file to use
fixation_parameters_file = os.path.normpath(os.path.join(erica_directory, 'Fixation_parameters', 'fixation_parameters_default.yaml'))
sys.stdout.write('Using parameter file: %s'%system_parameters_file)



save_folder_rd_output = os.path.normpath(os.path.join(erica_directory, 'Reaction_diffusion_outputs'))
if os.path.isdir(save_folder_rd_output) == 0:
    os.mkdir(save_folder_rd_output)

nlines_flyback  = myAOSLO.parameters['number_scan_lines_including_flyback'] - myAOSLO.parameters['number_scan_lines_excluding_flyback']



def load_isetbio_peaks(m):


    base_path = '/home/maria/Desktop/DatasetGeneratorERICA/ChocolateChip/ISETBio_Retinas'
    # find the retinas in the folder 
    data_path = f'Mosaic{str(m)}/0_2/'
    full_path = os.path.join(base_path, data_path)
    # full_path = '/home/maria/Desktop/DatasetGeneratorERICA/ChocolateChip/ISETBio_Retinas/0_0/'

    peaks_microns = pd.read_csv(os.path.join(full_path, 'all_cones_microns.csv')).to_numpy()
    print(f'\n\nPeak microns shape {peaks_microns.shape}')


    # Reading on one of the csv files in degrees or mirons 
    min_val_x = numpy.min(peaks_microns[:,0])
    min_val_y = numpy.min(peaks_microns[:,1])

    print(f'Minimum x {min_val_x}, Maximum x { numpy.max(peaks_microns[:, 0])}')
    print(f'Minimum y {min_val_y}, Maximum y { numpy.max(peaks_microns[:, 1])}')


    reshaped_peaks = peaks_microns[:, [1, 0]]

    print('Loading aperatures...')
    # Read in the cone aperature diameter in microns 
    peak_aperature = pd.read_csv(os.path.join(full_path, 'cone_aperature_microns.csv')).to_numpy()
    print(f'\n\nPeak aperature shape {peak_aperature.shape}')
   

    return reshaped_peaks, peak_aperature.T




def CreateSaveFolder():
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.mkdir(save_folder)

    
    # Save all associated metadata to the drive 
    meta_loc = os.path.join(save_folder, 'GenerationParameters')
    if os.path.isdir(meta_loc) == False:
        os.mkdir(meta_loc)
    else:    
        shutil.rmtree(meta_loc)
        os.mkdir(meta_loc)

    # Curcio, system parameters, reflectance parameters 

    shutil.copy(system_parameters_file, meta_loc)
    shutil.copy(fixation_parameters_file, meta_loc)
    shutil.copy(os.path.normpath(os.path.join(erica_directory, 'Reflectance_parameters', 'cone_reflectance_parameters.yaml')), meta_loc)

    return save_folder, meta_loc



def makeSaveFolder(save_folder, i, j):
    
    save_folderi = save_folder +  f'/ConeMosaic{i}/Session{j}'
    save_images_folder = os.path.normpath(os.path.join(save_folderi,'Images'))
    save_movement_folder = os.path.normpath(os.path.join(save_folderi,'Movement'))
    save_speed_folder = os.path.normpath(os.path.join(save_folderi,'Speed'))

    if os.path.isdir(save_folderi) == False:
        os.makedirs(save_folderi)
    if os.path.isdir(save_images_folder) == False:
        os.makedirs(save_images_folder)
    if os.path.isdir(save_movement_folder) == False:
        os.makedirs(save_movement_folder)
    if os.path.isdir(save_speed_folder) == False:
        os.makedirs(save_speed_folder)
    
    meta_data_file = os.path.normpath(os.path.join(save_folderi,'Meta_data.csv'))
    
    
    return save_folderi, save_images_folder, save_movement_folder, save_speed_folder, meta_data_file




def makeImage(aoslo, retina_params, eccentricity):
    
    myEyeMotion = fixation_simulation.Fixation(fixation_parameters_file, myAOSLO.time_resolution, subsamp=subsample_rate)

    microsaccade_angle = random.uniform(0, 360)
    microsaccade_angle *= (numpy.pi/180)

    microsaccade_amplitude = random.uniform(0, 30)
    microsaccade_amplitude = round(microsaccade_amplitude, 0)

    # The total time of the eye movement (seconds)
    # total_time = 1/30.
    total_time = aoslo.time_resolution * aoslo.n_samples

    drift_amp = None 
    # The number of samples in that time period
    n_samples_microsaccade = int(numpy.ceil(total_time / float(myAOSLO.time_resolution)))

    if microsaccade_amplitude > 0:
        # Log of the amplitude in degrees
        l = numpy.log10(microsaccade_amplitude/60.)

        # Estimate the maximum speed from the main sequence 
        ms = l + numpy.log10(myEyeMotion.parameters['Main_sequence_factor'])
        microsaccade_max_speed = 60 * 10**ms       
        
        # Generate the microsaccade
        microsaccade = fixation_simulation.microsaccade(microsaccade_amplitude, microsaccade_angle, microsaccade_max_speed, myAOSLO.time_resolution, n_samples=n_samples_microsaccade)

        # Calculate the velocity
        velocity =(microsaccade[:,1:] - microsaccade[:,:-1]) / myAOSLO.time_resolution

        radial_velocity = numpy.sqrt(velocity[1]**2 + velocity[0]**2)
        microsaccade_start = numpy.where(radial_velocity>radial_velocity.max()*0.1)[0][0]
        microsaccade_end = radial_velocity.shape[0] - numpy.where(radial_velocity[::-1]>radial_velocity.max()*0.1)[0][0]

        # https://pmc.ncbi.nlm.nih.gov/articles/PMC5082990/#:~:text=Ocular%20drift%20is%20commonly%20believed,during%20normal%20inter%2Dsaccadic%20fixation.
        # The amplitude of drift (arcminutes)

        if microsaccade_amplitude < 4:
            max_drift_amp = microsaccade_amplitude
        else:
            max_drift_amp = 4
        drift_amp = random.uniform(0, max_drift_amp)
       
        # The total time of the eye movement (seconds)
        total_time_drift = 2/30.

        # The number of samples in that time period
        n_samples_drift = int(numpy.ceil(total_time_drift / float(myAOSLO.time_resolution))) 
        # Generate drift
        drift_motion = fixation_simulation.drift(drift_amp, myAOSLO.time_resolution, n_samples_drift)
        microsaccade_copy = numpy.copy(microsaccade)
        
        drift_sequence_1 = drift_motion[:,:microsaccade_start]
        drift_sequence_1[0] = drift_sequence_1[0] - drift_sequence_1[0,-1]
        drift_sequence_1[1] = drift_sequence_1[1] - drift_sequence_1[1,-1]

        drift_sequence_1[0] = drift_sequence_1[0] + microsaccade_copy[0, microsaccade_start-1]
        drift_sequence_1[1] = drift_sequence_1[1] + microsaccade_copy[1, microsaccade_start-1]

        microsaccade_copy[:,:microsaccade_start] = drift_sequence_1

        drift_amp = random.uniform(0, max_drift_amp)

        # The total time of the eye movement (seconds)
        total_time_drift = 2/30.
        # The number of samples in that time period
        n_samples_drift = int(numpy.ceil(total_time_drift / float(myAOSLO.time_resolution)))
        # Generate drift
        drift_motion = fixation_simulation.drift(drift_amp, myAOSLO.time_resolution, n_samples_drift)


        drift_sequence_2 = drift_motion[:,:len(microsaccade[0,microsaccade_end:])]
        drift_sequence_2[0] = drift_sequence_2[0] - drift_sequence_2[0,0]
        drift_sequence_2[1] = drift_sequence_2[1] - drift_sequence_2[1,0]

        drift_sequence_2[0] = drift_sequence_2[0] + microsaccade_copy[0, microsaccade_end]
        drift_sequence_2[1] = drift_sequence_2[1] + microsaccade_copy[1, microsaccade_end]
        microsaccade_copy[:,microsaccade_end:] = drift_sequence_2

        microsaccade = microsaccade_copy[:,:aoslo.fast.shape[0]]

    else: 
        microsaccade = numpy.zeros((2, aoslo.slow.shape[0]))
        microsaccade_max_speed = 0.

    intensity_microsaccade, *_ = sampling.sampleRetina(retina_params, microsaccade, aoslo.fast, aoslo.slow, aoslo.parameters['pixel_size_arcmin_y_x'], aoslo.parameters['microns_per_degree'], eccentricity)
    fr_image_microsaccade = intensity_microsaccade.reshape((aoslo.parameters['number_scan_lines_including_flyback'], aoslo.parameters['number_pixels_per_line']*2))
    image_microsaccade = fr_image_microsaccade[:,:aoslo.parameters['number_pixels_per_line']]
    image_desinusoid_microsaccade = aoslo.desinusoid(image_microsaccade)
    img_dff = frame_generation.diffractionOnly(image_desinusoid_microsaccade[:-nlines_flyback], aoslo.parameters['pixel_size_arcmin_y_x'], aoslo.parameters['wavelength_m'], aoslo.parameters['pupil_diameter_m'], aoslo.parameters['pinhole_diameter_Airy_radii'])

    speed = numpy.zeros((microsaccade.shape))
    speed[0,1:] = (microsaccade[0,1:] - microsaccade[0,:-1]) / float(aoslo.time_resolution)
    speed[0,0] = speed[0,1] 
    
    speed[1,1:] = (microsaccade[1,1:] - microsaccade[1,:-1]) / float(aoslo.time_resolution)
    speed[1,0] = speed[1,1] 
    
    # Motion Y

    fr_image_microsaccade_y = microsaccade[0].reshape((aoslo.parameters['number_scan_lines_including_flyback'], aoslo.parameters['number_pixels_per_line']*2))
    image_microsaccade_y = fr_image_microsaccade_y[:,:aoslo.parameters['number_pixels_per_line']]
    motion_y = aoslo.desinusoid(image_microsaccade_y)
    
    fr_image_microsaccade_y = speed[0].reshape((aoslo.parameters['number_scan_lines_including_flyback'], aoslo.parameters['number_pixels_per_line']*2))
    image_microsaccade_y = fr_image_microsaccade_y[:,:aoslo.parameters['number_pixels_per_line']]
    speed_y = aoslo.desinusoid(image_microsaccade_y)
    
    # Motion X
    fr_image_microsaccade_x = microsaccade[1].reshape((aoslo.parameters['number_scan_lines_including_flyback'], aoslo.parameters['number_pixels_per_line']*2))
    image_microsaccade_x = fr_image_microsaccade_x[:,:aoslo.parameters['number_pixels_per_line']]
    motion_x = aoslo.desinusoid(image_microsaccade_x)
    fr_image_microsaccade_x = speed[1].reshape((aoslo.parameters['number_scan_lines_including_flyback'], aoslo.parameters['number_pixels_per_line']*2))
    image_microsaccade_x = fr_image_microsaccade_x[:,:aoslo.parameters['number_pixels_per_line']]
    speed_x = aoslo.desinusoid(image_microsaccade_x)
    movement = numpy.asarray([motion_y, motion_x])
    speed_img = numpy.asarray([speed_y, speed_x])
    

    return image_desinusoid_microsaccade[:-nlines_flyback], movement, speed_img, microsaccade_max_speed, microsaccade_angle, microsaccade_amplitude, drift_amp

def format_isetbio_retina(m, meta_data_file):
    
    peaks, peak_aperature = load_isetbio_peaks(m)

    # We want to end up with an array which follows the format:
    # [cone y pos, cone x pos, cone x width (stdev), cone y width (stdev), rotation angle (radians), reflectance]
    print('Generating rotation angles...')
    rotation_angles = numpy.random.uniform(0,2*numpy.pi, size=(peaks.shape[0]))
    print(f'Rotation angles {rotation_angles.shape}')

    print('Loading refelctance')
    print(cone_reflectance_parameters)

    normal_mean = cone_reflectance_parameters.get("normal_mean")
    normal_std = cone_reflectance_parameters.get("normal_std")


    reflectance = numpy.random.normal(numpy.float64(normal_mean), numpy.float64(normal_std), size=(peaks.shape[0]))
    print(reflectance.shape)
    print(reflectance)
    
   

    retina = numpy.zeros((6, peaks.shape[0]))

    retina[0, :] = peaks[:, 0].T
    retina[1, :] = peaks[:, 1].T
    retina[2, :] = peak_aperature[:peaks.shape[0]] / (2*1.96)
    retina[3, :] = peak_aperature[:peaks.shape[0]] / (2*1.96)
    retina[4, :] = rotation_angles[:].T
    retina[5, :] = reflectance[:].T

    print(retina.shape)
    
    name = os.path.join(meta_data_file, f'RetinalParameters_Mosaic{m}.csv')
    numpy.savetxt(name, retina, delimiter=',')

    
    return retina 




def main():

    save_folder, meta_loc = CreateSaveFolder()

    HowManyRepeats = 1
    totalIm = 0 
    num_mosaic = 4

    total_images = 70
    
    for m in range(num_mosaic):
        print('Mosaic ', m )
        retinal_mosaic_params_microns = format_isetbio_retina(m, meta_loc)

        for r in range(HowManyRepeats):
        
            save_folderi, save_images_folder, save_movement_folder, save_speed_folder, meta_data_file = makeSaveFolder(save_folder, m, r)

            myFile = open(meta_data_file, 'a')
            writer = csv.writer(myFile)
            writer.writerow(['File name','Direction (deg from positive x-axis)', 'Displacement (arcmin)','Max. speed (armin / second)', 'Drift Amplitude (arcmin)', 'Defocus Amplitude']) 
            
            for j in tqdm(range(total_images), position=0, leave=True):

                a = random.uniform(0, 2)
                psf, double_pass_psf = psf_generation_with_aberrations.ConstructPSF(pv_in=a, save_plots=save_folderi)


                fname = 'Image_%03i.bmp'%j

                #mg_dff, movement, speed_img, microsaccade_max_speed, microsaccade_angle, microsaccade_amplitude, drift_amp
                image_save, em, sp, ms, microsaccade_angle, microsaccade_dispalcement, drift_amp = makeImage(myAOSLO, retinal_mosaic_params_microns, eccentricity_fixation)


                processed_image = (image_save/image_save.max())

                downsample_w, downsample_h = image_save.shape
                downsampled_psf = cv2.resize(double_pass_psf, dsize=(downsample_h, downsample_h))
                # plt.close(); plt.cla(); plt.clf();
                # f = plt.figure(1, figsize=(10,5))
                # ax1 = f.add_subplot(111)
                # im1 = ax1.imshow(downsampled_psf, cmap='gray')
                # cb1 = plt.colorbar(im1, orientation='horizontal', ticks=numpy.linspace(0, 1.0, 3))
                # cb1.set_label('Downsampled PSF')
                # plt.savefig(os.path.join(save_folderi, f'Downsampled_PSF_{m}_{j}.png'))

                aberrated_image = convolve2d(processed_image, downsampled_psf, mode='full', boundary='fill', fillvalue=0)
                # print(aberrated_image.shape)
                x_min = int((aberrated_image.shape[0])/2 - (processed_image.shape[0]/2))
                x_max = int((aberrated_image.shape[0])/2 + (processed_image.shape[0]/2))
                
                y_min = int((aberrated_image.shape[1])/2 - (processed_image.shape[1]/2))
                y_max = int((aberrated_image.shape[1])/2 + (processed_image.shape[1]/2))
                # print(x_min, x_max, y_min, y_max)
                cropped_result = aberrated_image[x_min:x_max, y_min:y_max]


                Image.fromarray(ERICA_toolkit.normalise(cropped_result).astype(numpy.uint8)).save(os.path.normpath(os.path.join(save_images_folder, fname)))
                fits.writeto(os.path.normpath(os.path.join(save_movement_folder,fname.split('.')[0]+'.fits')), em, overwrite=True)
                fits.writeto(os.path.normpath(os.path.join(save_speed_folder,fname.split('.')[0]+'.fits')), sp, overwrite=True)

                writer.writerow([fname, microsaccade_angle, microsaccade_dispalcement, ms, drift_amp, a]) 
                
                totalIm += 1

        
    print('Done!')
    print(f'I made {totalIm} Images for {num_mosaic} Participants who were each imaged {HowManyRepeats} times.')


if __name__ == '__main__':  
    main()

