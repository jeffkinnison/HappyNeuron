import os
import glob
import dxchange
import numpy as np
import h5py

print ("Working Dir")
print (os.getcwd())


# cube = dxchange.read_tiff_stack('cube/z01.tiff',np.arange(1,25)) #raw data 8bit, change "256" to # of sections
cube1 = dxchange.read_tiff_stack('G0013_cube/z0001.tif',np.arange(1,100)) #raw data 8bit, change "256" to # of sections
#cube2 = dxchange.read_tiff_stack('inference_cube_02/z_0000.tif',np.arange(0,75)) #raw data 8bit, change "256" to # of sections


#cube3 = np.append(cube2,cube,axis=0)

# print (cube.shape)
# print ('Mean : '+str(cube.mean()))
# print ('Std : '+str(cube.std()))

print (cube1.shape)
print ('Mean : '+str(cube1.mean()))
print ('Std : '+str(cube1.std()))



#h5file = h5py.File('inference_data.h5', 'w')
#h5file.create_dataset('inference1',data=cube3)
h5file = h5py.File('G0013_cube_data.h5', 'w')
h5file.create_dataset('inference1',data=cube1)
h5file.close()

print ("Finished!! Goodbye!!")
