import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('file_handover.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

sphere_points = np.array(loaded_dict['Points']) #192*3   
det_list = np.array(loaded_dict['determinant'])  #192      d
jlwki_list = np.array(loaded_dict['jlwki'])      #192      d
forces_list = np.array(loaded_dict['forcemom']) #192*10*6  
sforce_list = np.array(loaded_dict['sum_force']) #192*10
smoment_list = np.array(loaded_dict['sum_moment']) #192*10


radi_max = 8
theta_max = 8
phi_max = 4


radi_list = np.zeros([ 192 ])
theta_list = np.zeros([ 192 ])
phi_list = np.zeros([ 192 ])
x = np.zeros([ 192 ])
y = np.zeros([ 192 ])
z = np.zeros([ 192 ])
arm_length=0.725

# count_=0
# for r in range(2,radi_max):
#     r=r*arm_length/radi_max
#     for theta in range(0,theta_max):
#         theta = theta*2*np.pi/theta_max
#         for phi in range(0,phi_max):            
#             phi = phi*np.pi/phi_max
#             radi_list[count_]=r
#             theta_list[count_]=theta
#             phi_list[count_]=phi
#             x[count_]=sphere_points[count_,0]
#             y[count_]=sphere_points[count_,1]
#             z[count_]=sphere_points[count_,2]
#             count_=count_+1

#det_list with x,y,z
#det_list with r,phi,theta
sforce_list_1 = np.sum( sforce_list, axis=1)
smoment_list_1 = np.sum( smoment_list, axis=1)


#mean = np.mean(det_list)
#print('det_list mean', mean)
#det_sort = np.sort(det_list)
#print('75th percentile', det_sort[48])

det_main_points_list = []
det_main_list = []
sforce_main_points_list = []
sforce_main_list = []
smoment_main_points_list = []
smoment_main_list = []
combine_points_list=[]

count_ = 0

for r in range(2,radi_max):
    r=r*arm_length/radi_max
    for theta in range(0,theta_max):
        theta = theta*2*np.pi/theta_max
        for phi in range(0,phi_max):            
            phi = phi*np.pi/phi_max
            radi_list[count_]=r
            theta_list[count_]=theta
            phi_list[count_]=phi
            
            if det_list[count_]>85.829:
            	if sforce_list_1[count_]<0.2172:
            		if smoment_list_1[count_]<0.045:
            			combine_points_list.append(sphere_points[count_])
            			print(sphere_points[count_])

            	# det_main_points_list.append(sphere_points[count_])
            	# det_main_list.append(det_list[count_])

            
            	# sforce_main_points_list.append(sphere_points[count_])
            	# sforce_main_list.append(sforce_list_1[count_])

            	# smoment_main_points_list.append(sphere_points[count_])
            	# smoment_main_list.append(smoment_list_1[count_])

            count_ = count_+1



dict_item = { 'combine' : combine_points_list}
f = open("file_handover_combine.pkl","wb")
pickle.dump(dict_item,f)
f.close()






# fig, axs = plt.subplots(3)


# limy=1
# plt.figure(figsize=[8 , 10])
# plt.suptitle('Force wrt cartesian')
# plt.subplot(3, 1, 1)
# plt.scatter(x,det_list)
# plt.ylim([0,limy])
# plt.axhline(y=det_sort[48], color='r', linestyle='-')
# plt.xlabel('x axis (m)')
# plt.subplot(3, 1, 2)
# plt.scatter(y,det_list)
# plt.ylim([0,limy])
# plt.axhline(y=det_sort[48], color='r', linestyle='-')
# plt.xlabel('y axis (m)')
# plt.ylabel('Force')
# plt.subplot(3, 1, 3)
# plt.scatter(z,det_list)
# plt.ylim([0,limy])
# plt.axhline(y=det_sort[48], color='r', linestyle='-')
# plt.xlabel('z axis (m)')
# #axs[3].plot(smooth(o_4,1))
# plt.savefig("det.png")


# limy=1
# plt.figure(figsize=[8, 10])
# #fig, axs = plt.subplots(3)
# plt.suptitle('Force wrt polar')
# plt.subplot(3, 1, 1)
# plt.scatter(radi_list,det_list)
# plt.ylim([0,limy])
# plt.axhline(y= det_sort[48], color='r', linestyle='-')
# plt.xlabel('radi (m)')
# plt.subplot(3, 1, 2)
# plt.scatter(theta_list,det_list)
# plt.ylim([0,limy])
# plt.axhline(y= det_sort[48], color='r', linestyle='-')
# plt.xlabel('theta (rad)')
# plt.ylabel('Force')
# plt.subplot(3, 1, 3)
# plt.scatter(phi_list,det_list)
# plt.ylim([0,limy])
# plt.axhline(y= det_sort[48], color='r', linestyle='-')
# plt.xlabel('phi (rad)')
# #axs[3].plot(smooth(o_4,1))
# #plt.set_ylabel('Determinant')
# plt.savefig("polar.png")



#print( jlwki_list )

#dict_item = { 'Points': points_sphere,'determinant' : det_list, 'jlwki' : jlwki_list, 'forcemom' : forces_list, 'sum_force': sforce_list, 'sum_moment': smoment_list}
