from nilearn import plotting

LIVER_PATH = '../data/Task03_Liver/'
LIVER_PREFIX = 'liver_'
LIVER_POSTFIX = '.nii.gz'
PROSTATE_PATH = '../data/Task05_Prostate/'
PROSTATE_PREFIX = 'prostate_'
PROSTATE_POSTFIX = '.nii.gz'
TRAINING = 'imagesTr/'
TEST = 'imagesTs/'
LABELS = 'labelsTr/'


def show_image(path):
    print(path)
    plotting.plot_img(path)
    plotting.show()

# Daten liegen im data Ordner, der sich im roo des Projektes befindet.
def main():
    print("Gude!")
    show_image(LIVER_PATH + TRAINING + 'liver_0.nii')

main()