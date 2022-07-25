import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D


with open('file_handover_2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

sphere_points = np.array(loaded_dict['Points']) #192*3   
det_list = np.array(loaded_dict['determinant'])  #192      d
#jlwki_list = np.array(loaded_dict['jlwki'])      #192      d
#forces_list = np.array(loaded_dict['forcemom']) #192*10*6  
sforce_list = np.array(loaded_dict['sum_force']) #192*10
smoment_list = np.array(loaded_dict['sum_moment']) #192*10
check_mask_list = np.array(loaded_dict['reach_check'])
f_moment = np.array(loaded_dict['fobj_moment'])
f_angle_list = np.array(loaded_dict['fobj_angle'])

f_moment_mean = np.mean(f_moment)
f_moment[np.argmax(f_moment)]=f_moment_mean

#print('sphere_points', sphere_points)
#print('sforce_list', sforce_list)
#print('smoment_list', smoment_list)
print('check_mask_list', check_mask_list)
#print('f_moment', f_moment)
#print('f_angle_list', f_angle_list)
print('check_mask reached', np.sum(check_mask_list) )

f_angle_obj = f_angle_list/np.max(f_angle_list)
f_moment_obj = f_moment/np.max(f_moment)
f_det_obj = det_list/np.max(det_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

img = ax.scatter(sphere_points[:,0], sphere_points[:,1], sphere_points[:,2], c=check_mask_list, cmap=plt.hot())
#cmap=plt.hot(), 'seismic'
fig.colorbar(img)
plt.show()