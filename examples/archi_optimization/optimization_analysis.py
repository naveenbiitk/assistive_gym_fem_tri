import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('file_modified_showscreen.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

#{ 'Points': point_optimized,' score ' : score  }
print(loaded_dict)
points = np.array(loaded_dict['Points']) #192*3   
score = np.array(loaded_dict[' score '])  #192      d



# score  size (256,6)
print('Evaluation analysis:  (lower means better)')
print('score_wrt_head'  , score[0]) 
print('score_wrt_neck'  , score[1]) 
print('score_wrt_chest'  , score[2]) 
print('score_wrt_waist'  , score[3]) 
print('score_wrt_stomach'  , score[4]) 
print('score_wrt_left_shoulder'  , score[5]) 
print('score_wrt_right_shoulder'  , score[6]) 
print('score_wrt_left_elbow'  , score[7]) 
print('score_wrt_right_elbow'  , score[8]) 


parts = ['Head', 'Neck', 'Chest', 'Waist', 'Stomach', 'Left_Should', 'Right_Should', 'Left_elbow', 'Right_elbow' ]
func = [ score[0],  score[1],  score[2],  score[3],  score[4],  score[5],  score[6],  score[7],  score[8]]
# courses = list(data.keys())
# values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(parts, func, color ='maroon',
        width = 0.4)

plt.ylim([0, 500])
 
plt.xlabel("Body parts")
plt.ylabel("Average Obj. function score")
plt.title("Optimal Show_screen frame")
plt.show()