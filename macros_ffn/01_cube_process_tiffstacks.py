import os
import glob
import dxchange
import numpy as np
import h5py

print ("Working Dir")
print (os.getcwd())


cube = dxchange.read_tiff_stack('cube02/z01.tif',np.arange(1,50)) #raw data 8bit, change "256" to # of sections
labels = dxchange.read_tiff_stack('labels02/l01.tif',np.arange(1,50)) #label "ground truth" uint 8 or 32

print ('Cube Properties!')
print (cube.dtype)
print (cube.shape)

print ('Mean : '+str(cube.mean()))
print ('Std : '+str(cube.std()))

print ('Labels Properties!')
print (labels.dtype)
print (labels.shape)


print ('Ids Properties!')
ids = np.unique(labels,return_counts=1)
print (ids)

#raf added here to pad256
# cube  = np.pad(cube,((115,116),(0,0),(0,0)),'reflect')
# labels  = np.pad(labels,((115,116),(0,0),(0,0)),'reflect')

# print ('Cube Properties!')
# print (cube.dtype)
# print (cube.shape)
# print (cube.mean(),cube.std())

# print ('Labels Properties!')
# print (labels.dtype)
# print (labels.shape)
# print (labels.mean())



h5file = h5py.File('training_data_02.h5', 'w')
h5file.create_dataset('image',data=cube)
h5file.create_dataset('labels',data=labels)
h5file.close()

print ("Finished!! Goodbye!!")
