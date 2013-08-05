import numpy
from matplotlib import pylab
from PIL import Image
from conv import f

img = Image.open(open('./3wolfmoon.jpg'))
img = numpy.asarray(img, dtype='float64')/256

img_ = img.swapaxes(0,2).swapaxes(1,2).reshape(1,3,584,487)
#img_ = img.swapaxes(0,2).swapaxes(1,2).reshape(1,3,128,128)
filtered_img = f(img_)

pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0,1,:,:])
#pylab.savefig('lena_conv.jpg')
#pylab.savefig('conv.jpg')
pylab.show('./3wolfmoon.jpg')

