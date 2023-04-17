import numpy as np
import matplotlib.pyplot as plt

arr = np.load('evaluation_handover.npy')
#arr = np.load('evaluation_showscreen.npy')


# arr  size (256,6)
print('Evaluation analysis:  (lower means better)')
print('point_wrt_head'  ,np.mean(arr[:,0]) )
print('point_wrt_neck'  ,np.mean(arr[:,1]) )
print('point_wrt_chest'  ,np.mean(arr[:,2]) )
print('point_wrt_waist'  ,np.mean(arr[:,3]) )
print('point_wrt_stomach'  ,np.mean(arr[:,4]) )
print('point_wrt_left_shoulder'  ,np.mean(arr[:,5]) )
print('point_wrt_right_shoulder'  ,np.mean(arr[:,6]) )
print('point_wrt_left_elbow'  ,np.mean(arr[:,7]) )
print('point_wrt_right_elbow'  ,np.mean(arr[:,8]) )


parts = ['Head', 'Neck', 'Chest', 'Waist', 'Stomach', 'Left_Should', 'Right_Should', 'Left_elbow', 'Right_elbow' ]
func = [np.mean(arr[:,0]), np.mean(arr[:,1]), np.mean(arr[:,2]), np.mean(arr[:,3]), np.mean(arr[:,4]), np.mean(arr[:,5]), np.mean(arr[:,6]), np.mean(arr[:,7]), np.mean(arr[:,8])]
# courses = list(data.keys())
# values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(parts, func, color ='maroon',
        width = 0.4)
 
plt.xlabel("Body parts")
plt.ylabel("Average Obj. function score")
plt.title("Optimal Handover frame")
plt.show()