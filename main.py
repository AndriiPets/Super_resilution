
import numpy as np
from PIL import Image
from ISR.models import RDN
import shutil



img = Image.open('test_images/section8-image.png')
lr_img = np.array(img)

rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')

def predict():
    sr_img = rdn.predict(lr_img)
    return Image.fromarray(sr_img)



def main():
    img = predict()
    
    img.save('output.png', format='png')           


if __name__ == '__main__':
    main() 