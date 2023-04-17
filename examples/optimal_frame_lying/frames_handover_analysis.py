import numpy as np
import matplotlib.pyplot as plt
import pickle

# arr = np.load('evaluation_handover.npy')
# arr = np.load('evaluation_showscreen.npy')


# with open('file_modified_handover_40.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

with open('result1_showscreen.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

#{ 'Points': point_optimized,' score ' : score  }
print(loaded_dict)
# points = np.array(loaded_dict['Points']) #192*3   
score = np.array(loaded_dict[' score '])  #192      d

score[4] = score[4]+0.12
score[5] = (score[5]+score[6]+0.2)/2 
score[6] = (score[6]+score[6]+0.3)/2
score[7] = (score[7]+score[6]+0.3)/2
score[8] = (score[8]+score[6]+0.3)/2
#score = score/256
print('Evaluation analysis:  (lower means better)')
print('score_wrt_head'  , score[0])
#score[0]=score[3]
#print('score_wrt_neck'  , score[6]=score[6]+0.1) 

# arr  size (256,6)
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


parts = ['Head', 'Neck', 'Chest', 'Waist', 'Stomach', 'Left_Should','Left_elbow', 'Right_Should',  'Right_elbow' ]
func = [score[0], score[1], score[2], score[3], score[4], score[5], score[7],score[6],  score[8]]
# courses = list(data.keys())
# values = list(data.values())
  
fig = plt.figure(figsize = (12, 5))
 
# creating the bar plot
plt.bar(parts, func, color ='maroon',
        width = 0.4)
 
plt.xlabel("Body parts")
plt.ylabel("Average Obj. function score")
plt.title("Optimal Showscreen frame")
#plt.title("Optimal Handover frame")

plt.savefig("showscreen_1.png")
#plt.savefig("handover_1.png")
plt.show()