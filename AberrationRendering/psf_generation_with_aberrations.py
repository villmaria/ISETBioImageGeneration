import os 
import math
import cmath 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

DIRNAME = os.path.split(__file__)[0]
save_plots = os.path.join(DIRNAME, 'plots')

def ZernikePolar(coefficients, r, u):
    '''
    Calculate the Zernike polynomial in polar coordinates
    '''

    # print(f'Coefficients: {coefficients}')
    Z = coefficients
    Z1 = Z[0] * 1
    
    # Z2 = Z[1] * 2*r*np.cos(u)
    Z2 = Z[1] * 2 * np.multiply(r, np.cos(u))

    # Z3 = Z[2] * 2*r*np.sin(u)
    Z3 = Z[2] * 2 * np.multiply(r, np.sin(u))

    # Z4 = Z[3] * np.sqrt(3)*(2*r**2 - 1)
    Z4 = Z[3] * np.sqrt(3) * (2 * np.square(r) - 1)


    ZW = Z1 + Z2 + Z3 + Z4
    # print(Z1, Z2, Z3, Z4)
    # print(ZW.shape, ZW)
    print(f'Shape ZW {ZW.shape}')
    print(f'Sum ZW {np.sum(ZW)}')

    return ZW

def PupilSize(n, pupil_diameter, lamba, pixel_rad, det_size):
    '''
    Calculate the size of the pupil size raidus
    n: refractive index of the medium
    pupil_diameter: diameter of the pupil
    lamba: wavelength of the light
    pixel_rad: radius of the pixel
    det_size: size of the detector

    return: radius of the pupil
    '''

    # Cut off frequency in rad^-1
    nu_cutoff = (n*pupil_diameter)/(lamba)
    # Sampling interval in rad^-1
    delta_nu = 1/(det_size*pixel_rad)
    #Radius of the pupil in pixels
    rpupil = nu_cutoff/(2*delta_nu)

    print(f'R PUPIL {rpupil}')

    return rpupil


def Phase(coefficients, rpupil):
    '''
    Define wavefront of exit pupil
    '''
    r = 1
    x = np.linspace(-r, r, int(2*rpupil))
    # print(x.shape, x)
    y = np.linspace(-r, r, int(2*rpupil))
    X, Y = np.meshgrid(x, y, indexing='ij')
    # print(Y.shape, Y)
    R = np.sqrt(X**2 + Y**2)
    # print(R.shape, R)
    theta = np.arctan2(Y, X)
    # print(theta.shape, theta)
    Z = ZernikePolar(coefficients, R, theta)
    

    # temp = np.zeros(())
    repalce = np.where(R > 1)
    Z[repalce] = 0
    # print(Z[repalce].shape)
    # print(Z.shape, Z)
    # print(np.sum(Z))

    print(f'Count non zero Phase {np.count_nonzero(Z)}')
    print(f'Sum Phase {np.sum(Z)}')

    return Z

def Center(coefficients, det_size, rpuil):
    '''
    '''
    A = np.zeros((int(det_size), int(det_size)))

    part1 = int(np.floor( det_size / 2  - rpuil + 1))
    part2 = int(np.floor( det_size / 2  + rpuil))
    part3 = int(np.floor( det_size / 2  - rpuil + 1 ))
    part4 = int(np.floor( det_size / 2  + rpuil))
    
    Z = Phase(coefficients, rpuil)
    A[part1: part2, part3: part4] = Z


    print(f'Count non zero Z {np.count_nonzero(Z)}')
    print(f'Count non zero A {np.count_nonzero(A)}')
    print(f'Sum Z {np.sum(Z)}')
    print(f'Sum A {np.sum(A)}')


    # print(f'Num 1 {np.floor( det_size / 2  - rpuil + 1)}')
    # print(f'Num2 : {np.floor( det_size / 2  + rpuil)}')
    # print(f'Num3 : {np.floor( det_size / 2  - rpuil + 1 )}')
    # print(f'Num4 : { np.floor( det_size / 2  + rpuil)}')

    return A

def Mask(rpupil, det_size):
    
    r = 1 
    x = np.linspace(-r, r, int(2*rpupil))
    y = np.linspace(-r, r, int(2*rpupil))

    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    M = 1*((np.cos(theta)**2) + (np.sin(theta)**2))
    M[R > 1] = 0

    mask = np.zeros((int(det_size), int(det_size)))
    # mask[int(det_size/2 - rpupil+1): int(det_size/2 + rpupil), int(det_size/2 - rpupil+1):int(det_size/2 + rpupil)] = M

    part1 = int(np.floor( det_size / 2  - rpupil + 1))
    part2 = int(np.floor( det_size / 2  + rpupil))
    part3 = int(np.floor( det_size / 2  - rpupil + 1 ))
    part4 = int(np.floor( det_size / 2  + rpupil))
    mask[part1: part2, part3: part4] = M


    count_non_zero = np.count_nonzero(M)
    print(f'Count non zero {count_non_zero}')
    mask_non_zero = np.count_nonzero(mask)
    print(f'Mask non zero {mask_non_zero}')
    
    idx_non_zero_M  = np.nonzero(M)
    print(f'M non zero {idx_non_zero_M}')
    idx_non_zero_mask  = np.nonzero(mask)
    print(f'Mask non zero {idx_non_zero_mask}')

    print(f'Sum M = {np.sum(M)}')
    print(f'Sum Mask = {np.sum(mask)}')

    return mask


def ComplexPupil(A, mask):
    '''
    Calculate the complex pupil function
    '''
    
    abbe = np.zeros((A.shape[0], A.shape[1]), dtype=complex)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            abbe[r, c] = cmath.exp(A[r, c]*1j)

    abbe_z =  np.zeros((abbe.shape[1], abbe.shape[1]), dtype=complex)
    
    abbe_z = np.multiply(abbe, mask)

    print(f'None zero Abbe_z {np.count_nonzero(abbe_z), np.sum  (abbe_z) }')

    # print(np.where(abbe_z > 0))

    return abbe_z
    

def PSF(complex_pupil, save_plots):
    '''
    Calculate the point spread function
    '''
    dpi = 140

    amp_psf_orig = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(complex_pupil)))
    int_psf_orig = np.square(np.abs(amp_psf_orig))

    int_psf_org_max = np.max(int_psf_orig)
    curr_psf = int_psf_orig/int_psf_org_max
    # print(curr_psf.shape, curr_psf)

    # f = plt.figure(1)

    # ax1 = f.add_subplot(111)
    # im1 = ax1.imshow(int_psf_orig, cmap='gray')
    # cb1 = plt.colorbar(im1, orientation='horizontal', ticks=np.linspace(0, 1.0, 3))
    # cb1.set_label('Original PSF')
    # plt.savefig(os.path.join(save_plots, 'Original_PSF.tiff'),  format='tiff')
    # # Image.fromarray(im1).save(os.path.join(save_plots, 'Original_PSF.tiff'))
    # # plt. axis('off') 
    # plt.savefig(os.path.join(save_plots, 'Original_PSF.tiff'),  format='tiff', dpi=dpi, bbox_inches = 'tight')



    # plt.cla(); plt.clf(), plt.close()
    # f = plt.figure(1)
    # ax2 = f.add_subplot(111)
    # im2 = ax2.imshow(curr_psf, cmap='gray')
    # cb2 = plt.colorbar(im2, orientation='horizontal', ticks=np.linspace(0, 1.0, 3))
    # cb2.set_label('Normalised PSF')

    # # Image.fromarray(im2).save(os.path.join(save_plots, 'Normalised_PSF.tiff'))
    # # plt. axis('off') 
    # plt.savefig(os.path.join(save_plots, 'Normalised_PSF.tiff'),  format='tiff', dpi=dpi, bbox_inches = 'tight')

    
    print(f'Single Pass None Zero {np.count_nonzero(int_psf_orig), np.sum(int_psf_orig)}')
    print(f'Max {int_psf_org_max}')
    print(f'Sinlge Pass None Zero normal {np.count_nonzero(curr_psf), np.sum(curr_psf)}')
    

    return int_psf_orig, curr_psf, int_psf_org_max


def ConstructPSF(pv_in, save_plots):

    # FOV radius in um 
    FOV_rad = 187
    # Pixel size in um
    pixel_size_PSF  = 0.2
    # Focal length in mm (of eye) 1.7e4
    foc_length = 2e4
    # Size of each pixel in radians 
    pizel_size_PSF_rad = pixel_size_PSF/foc_length
    # Number of pixels in PSF image

    pixel_num = ((FOV_rad*2)/ pixel_size_PSF) + 1
    # print(f'pixel num {pixel_num}')

    # Pupil diameter 
    pupil_diam = 4.0e3

    # Wavelength of light in um
    lamba = 0.83
    norm_thresh = 0.1
    zero_thresh = 5e-04

    # Amplitude of the aberrations in unit of wavelenght (distance) wavelenght*unit = microns of aberrations 
    pv_amp = pv_in
    pv_rms = 3.51
    pupil_area = ((pupil_diam/(2*1e3)) **2) * np.pi
    rms_amp_defoc = pv_amp * (lamba/pv_rms)
    diopters = 4 * np.pi * np.sqrt(3) * (rms_amp_defoc/pupil_area)

    print(f'Aberration amplitude: {pv_amp}')
    print(f'Aberration RMS: {pv_rms}')
    print(f'Aberration in diopters: {diopters}')
    n_water = 1.33
    NA = n_water * (pupil_diam/(2*foc_length))
    int_PSF_cutoff  = (0.61 * lamba)/NA
    FOV_int_psf = 10
    pixel_int_PSF = FOV_int_psf/pixel_size_PSF
    FOV_img_PSF = 5
    pixel_img_PSF = FOV_img_PSF/pixel_size_PSF

    coeff = pv_amp * lamba * ((2*np.pi)/lamba) / pv_rms
    coefficients = np.zeros(4)
    coefficients[-1] = coeff
    
    pixel_num_pupil_rad = PupilSize( n_water, pupil_diam, lamba, pizel_size_PSF_rad, pixel_num )
    sim_phase = Center(coefficients, pixel_num, pixel_num_pupil_rad)
    pupil_mask = Mask(pixel_num_pupil_rad, pixel_num)

    # dpi = 140 
    # f = plt.figure(2)
    # ax1 = f.add_subplot(111)
    # im1 = ax1.imshow(sim_phase, interpolation='nearest',cmap='gray')
    # cb1 = plt.colorbar(im1, orientation='horizontal', ticks=np.linspace(0, 1.0, 3))
    # cb1.set_label(r'Amplitude function, p(r,$\theta$)' + '\n(normalised units of transmission)')
    # plt.savefig(os.path.join(save_plots, 'Simphase.tiff'),  format='tiff')
    # Image.fromarray(im1).save(os.path.join(save_plots, 'Simphase.tiff'))

    # plt. axis('off') 
    # plt.savefig(os.path.join(save_plots, 'Simphase.tiff'),  format='tiff', dpi=dpi, bbox_inches = 'tight')



    pupil_com = ComplexPupil(sim_phase, pupil_mask)
    # Original PSF, Normalised PSF, Max in PSF pre-norm
    int_psf_orig, curr_psf, int_psf_org_max = PSF(pupil_com, save_plots)



    img_lat_PSF_sat = int_psf_orig
    img_lat_PSF_sat[img_lat_PSF_sat > norm_thresh * int_psf_org_max] = norm_thresh * int_psf_org_max

    int_lat_PSF_sat_max = np.max(img_lat_PSF_sat)
    int_lat_PSF_sat_norm = img_lat_PSF_sat/int_lat_PSF_sat_max

    # plt.cla(); plt.clf(), plt.close()
    # f = plt.figure(1)
    # ax1 = f.add_subplot(111)
    # im1 = ax1.imshow(int_lat_PSF_sat_norm, cmap='gray')
    # cb1 = plt.colorbar(im1, orientation='horizontal', ticks=np.linspace(0, 1.0, 3))
    # cb1.set_label('Saturated image intensity')
    # plt.savefig(os.path.join(save_plots, 'SatImIn.tiff'),  format='tiff')
    # # Image.fromarray(im1).save(os.path.join(save_plots, 'SatImIn.tiff'))

    # plt. axis('off') 
    # plt.savefig(os.path.join(save_plots, 'SatImIn.tiff'),  format='tiff', dpi=dpi, bbox_inches = 'tight')



    img_lat_PSF_org = np.square(curr_psf)
    img_lat_PSF_orig_max = np.max(img_lat_PSF_org)
    img_lat_PSF_norm = img_lat_PSF_org/ img_lat_PSF_orig_max

    
    print(f'Double Pass None Zero {np.count_nonzero(img_lat_PSF_org), np.sum(img_lat_PSF_org)}')
    print(f'Max {img_lat_PSF_orig_max}')
    print(f'Double Pass None Zero normal {np.count_nonzero(img_lat_PSF_norm), np.sum(img_lat_PSF_norm)}')
    
    # plt.cla(); plt.clf(), plt.close()
    # f = plt.figure(1)
    # ax1 = f.add_subplot(111)
    # im1 = ax1.imshow(img_lat_PSF_norm, cmap='gray')
    # cb1 = plt.colorbar(im1, orientation='horizontal', ticks=np.linspace(0, 1.0, 3))
    # cb1.set_label('Double Pass PSF Normalised')
    # plt.savefig(os.path.join(save_plots, 'DoublePassPSFNorm.tiff'),  format='tiff')
    # Image.fromarray(im1).save(os.path.join(save_plots, 'DoublePassPSFNorm.tiff'))

    # plt. axis('off') 
    # plt.savefig(os.path.join(save_plots, 'DoublePassPSFNorm.tiff'),  format='tiff', dpi=dpi, bbox_inches = 'tight')



    
    return curr_psf, img_lat_PSF_norm

if __name__ == '__main__':

    amplitude = 0.5
    debug_images = 'debug'
    save_plots = os.path.join('./', debug_images)   
    save_defocus = [amplitude]
    amp_meta = os.path.join(save_plots, 'DefocusAmplitude.txt')


    print(f'Current working directory: {DIRNAME}')
    print(f'Saving plots to: {save_plots}')
    _, _ = ConstructPSF(pv_in=amplitude, save_plots=save_plots)

