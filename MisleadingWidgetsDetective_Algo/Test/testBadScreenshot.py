from PIL import Image
from PIL import ImageFilter

image = Image.open('/data/DroidBot_Epoch/raw_data/input_data/org.kore.kolabnotes.android_101.apk/images/view_13bf9a32ff7f777911ec0964182d169b.png','r')
im = image.filter(ImageFilter.CONTOUR)
im.save('/home/crepuscule/Desktop/6.png')

