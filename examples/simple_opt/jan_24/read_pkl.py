import pickle
import numpy as np

f_base_name = 'cane_frame_analysis_jan_24_seq'
f_ext = '.pkl'
Points_arr = np.zeros([9,3])
score_arr = np.zeros([9])
frame_i_arr = np.zeros([9])
for f_var in range(9):
    f_final = f_base_name+str(f_var)+f_ext
    with open(f_final, 'rb') as f:
        data = pickle.load(f)
    
    Points_arr[f_var] = data['Points'] 
    score_arr[f_var] = data[' score ']
    frame_i_arr[f_var] = data['frame_i']

#print('frame:',frame_i_arr)
print("score",score_arr/max(score_arr))

print('Evaluation analysis:  (lower means better)')
# print('point_wrt_head'  ,np.mean(arr[:,0]) )
# print('point_wrt_neck'  ,np.mean(arr[:,1]) )
# print('point_wrt_chest'  ,np.mean(arr[:,2]) )
# print('point_wrt_waist'  ,np.mean(arr[:,3]) )
# print('point_wrt_stomach'  ,np.mean(arr[:,4]) )
# print('point_wrt_left_shoulder'  ,np.mean(arr[:,5]) )
# print('point_wrt_right_shoulder'  ,np.mean(arr[:,6]) )
# print('point_wrt_left_elbow'  ,np.mean(arr[:,7]) )
# print('point_wrt_right_elbow'  ,np.mean(arr[:,8]) )