# Enhanced_Nodule_Preprocessing
Enhancing Lung Nodule Preprocessing using rescale intensity , gamma correction , sigmoid correction , logarithmic correction and histogram equalization methods

The below code processed in enhnacement

noduleimages = np.load(datafolder + '/noduleimages.npy')
rescale_intensity_output=skimage.exposure.rescale_intensity(noduleimages,in_range=(0, 255))
sigmoid_correction=skimage.exposure.adjust_sigmoid(rescale_intensity_output)
gamma_correction=skimage.exposure.adjust_gamma(sigmoid_correction)
logarithmic_correction=skimage.exposure.adjust_log(gamma_correction)
enhanced_nodule_image = exposure.equalize_hist(logarithmic_correction)
