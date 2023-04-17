

self.henry_joints = self.pose_reindex()


    def pose_reindex(self):

        smpl_pose_jt_1 =  self.load_smpl_model()
        self.human_pos_offset, self.human_orient_offset = self.human.get_base_pos_orient()

        joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_2 = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        smpl_pose_jt_2 = smpl_pose_jt_2.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        smpl_pose_jt = smpl_pose_jt_2.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        
        return self.smpl_agym_map(smpl_pose_jt)



	def load_smpl_model(self):
        
        directory='/nethome/nnagarathinam6/hrl_git/assistive-gym-fem/assistive_gym/envs/assets'
        model_folder = os.path.join(directory, 'smpl_models')
        model = smplx.create(model_folder, model_type='smpl', gender=self.human.gender)

        print('Smple sample', self.sample_pkl)
        with open(self.f_name, 'rb') as handle:
            data1 = pickle.load(handle)

        df = torch.Tensor(data1['body_pose'])
        dt = torch.reshape(df, (1, 23, 3))
        db = dt[:,:21,:]
        
        orient_tensor = torch.Tensor(data1['global_orient'])
        self.orient_body = orient_tensor.numpy()

        body_pose = np.zeros((1,23*3))
        self.smpl_body_pose = dt[0].numpy()

        output = model(betas=torch.Tensor(data1['betas']), body_pose=data1['body_pose'], return_verts=True)
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=torch.Tensor(body_pose), return_verts=True)

        joints = output.joints.detach().cpu().numpy().squeeze()
        #print('output joint', joints.shape, joints)
        return joints
            
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=db, return_verts=True)
            
##--------------------------------------


    def convert_smpl_body_to_gym(self):

        smpl_pose_jt_1 = self.smpl_body_pose  #self.smpl_agym_map(self.smpl_body_pose)
        
        print('orient_body', self.orient_body)
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        
        smpl_bp = smpl_pose_jt_1
        # joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        # smpl_pose_jt_2 = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        # smpl_pose_jt_2 = smpl_pose_jt_2.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        # smpl_pose_jt = smpl_pose_jt_2.dot(R.from_euler('z', 180, degrees=True).as_matrix())

        

        self.human.set_base_pos_orient([0, 0.0, 0.80], [-np.pi/2, self.orient_body[0,2],0])

        opts_joints = [ self.human.j_head_x, self.human.j_head_y, self.human.j_head_z,self.human.j_neck,
                        self.human.j_upper_chest_x, self.human.j_upper_chest_y, self.human.j_upper_chest_z,
                        self.human.j_chest_x, self.human.j_chest_y, self.human.j_chest_z,
                        self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z ]

        cor_angles = [  smpl_bp[14,0],smpl_bp[14,1],smpl_bp[14,2],smpl_bp[11,0],
                        smpl_bp[8,0],smpl_bp[8,1],smpl_bp[8,2],
                        smpl_bp[5,0],smpl_bp[5,1],smpl_bp[5,2],
                        smpl_bp[2,0],smpl_bp[2,1],smpl_bp[2,2] ]

        self.human.set_joint_angles(opts_joints, cor_angles)
        #print('stomach cor_angles', np.array(cor_angles)*180/3.14)
        

        opts_joints = [ self.human.j_left_hip_x, self.human.j_left_hip_y, self.human.j_left_hip_z, self.human.j_left_knee ,
                        self.human.j_left_ankle_x, self.human.j_left_ankle_y, self.human.j_left_ankle_z ]

        cor_angles = [smpl_bp[0,0],smpl_bp[0,1],smpl_bp[0,2],smpl_bp[3,0],
                      smpl_bp[6,0],smpl_bp[6,1],smpl_bp[6,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)
        #print('right leg cor_angles', np.array(cor_angles)*180/3.14)

        opts_joints = [ self.human.j_right_hip_x, self.human.j_right_hip_y, self.human.j_right_hip_z, self.human.j_right_knee, 
                        self.human.j_right_ankle_x, self.human.j_right_ankle_y, self.human.j_right_ankle_z]

        cor_angles = [smpl_bp[1,0],smpl_bp[1,1],smpl_bp[1,2],smpl_bp[4,0],
                      smpl_bp[7,0],smpl_bp[7,1],smpl_bp[7,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)
        #print('left leg cor_angles', np.array(cor_angles)*180/3.14)

        opts_joints = [ self.human.j_left_pecs_x, self.human.j_left_pecs_y, self.human.j_left_pecs_z ,
                        self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z ,
                        self.human.j_left_elbow, self.human.j_left_forearm ]
        

        cor_angles_left = [  -smpl_bp[12,2]-1.57,smpl_bp[12,1],smpl_bp[12,0],
                             ((-smpl_bp[15,2])-1.57),smpl_bp[15,1],-(smpl_bp[15,0]),
                             smpl_bp[17,2],smpl_bp[19,1]]

        #cor_angles_left = [0,0,0,-3.14,0,0,0,0]
        #shoulder x didn't make any difference

        self.human.set_joint_angles(opts_joints, cor_angles_left)
        #print('left hand cor_angles', np.array(cor_angles_left)*180/3.14)

        opts_joints = [ self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z ,
                        self.human.j_right_shoulder_x, self.human.j_right_shoulder_y, self.human.j_right_shoulder_z,
                        self.human.j_right_elbow, self.human.j_right_forearm]
        

        
        ck = smpl_bp[16,0] if smpl_bp[16,0]<0 else smpl_bp[16,0]+1.57

        cor_angles_right = [smpl_bp[13,2],smpl_bp[13,1],smpl_bp[13,0],
                            1.57-smpl_bp[16,2],smpl_bp[16,1],ck,
                            -smpl_bp[18,2],smpl_bp[20,1]]


        self.human.set_joint_angles(opts_joints, cor_angles_right)
        print('right hand cor_angles', np.array(cor_angles_right)*180/3.14)



    def smpl_agym_map(self, smpl_pose_jt):

        agym_jt_smpl = np.zeros((20,3))

        agym_jt_smpl[0,:] = smpl_pose_jt[15,:] + np.array([ 0.00000000e+00 , 3.72529017e-09, -3.14101279e-02]) 
        agym_jt_smpl[1,:] = smpl_pose_jt[12,:] + np.array([0.00000000e+00 ,1.86264505e-09 ,7.45058060e-09]) 
        agym_jt_smpl[2,:] = smpl_pose_jt[3,:]  + np.array([ 0.00000000e+00 , 3.72529028e-09 ,-2.98023224e-08])
        agym_jt_smpl[3,:] = smpl_pose_jt[0,:]  + np.array([0., 0., 0.])
        agym_jt_smpl[4,:] = smpl_pose_jt[6,:]  + np.array([0.00000000e+00 ,3.72529026e-09 ,2.98023224e-08])
        agym_jt_smpl[5,:] = smpl_pose_jt[9,:]  + np.array([ 0.00000000e+00 ,-5.55111512e-17 ,-7.45058060e-09])
        #arms
        agym_jt_smpl[6,:] = smpl_pose_jt[17,:] + np.array([0.00000000e+00 ,2.79396764e-09 ,0.00000000e+00])
        agym_jt_smpl[7,:] = smpl_pose_jt[16,:] + np.array([ 0.00000000e+00 , 2.79396763e-09 ,-2.98023224e-08])
        agym_jt_smpl[8,:] = smpl_pose_jt[19,:] + np.array([ -0.00457314, -0.0248985 ,  0.02630745])
        agym_jt_smpl[9,:] = smpl_pose_jt[18,:] + np.array([-0.00560603, -0.03206245  ,0.03407168])
        agym_jt_smpl[10,:] = smpl_pose_jt[21,:] + np.array([-0.02053231 ,-0.02895589 ,-0.0012251 ]) 
        agym_jt_smpl[11,:] = smpl_pose_jt[20,:] + np.array([ 0.01074898, -0.02654349  ,0.00240007]) 
        agym_jt_smpl[12,:] = smpl_pose_jt[14,:] + np.array([0.00000000e+00 ,3.72529022e-09 ,7.45058060e-08]) 
        agym_jt_smpl[13,:] = smpl_pose_jt[13,:] + np.array([0.00000000e+00, 3.72529022e-09 ,2.98023224e-08]) 
        #legs
        agym_jt_smpl[14,:] = smpl_pose_jt[5,:] + np.array([ 0.00000000e+00 , 3.72529040e-09 ,-6.24269247e-03])
        agym_jt_smpl[15,:] = smpl_pose_jt[4,:] + np.array([ 0.00000000e+00,  1.86264525e-09 ,-4.67756391e-03])
        agym_jt_smpl[16,:] = smpl_pose_jt[2,:] + np.array([0.00000000e+00 ,3.72529033e-09 ,7.57336617e-04])
        agym_jt_smpl[17,:] = smpl_pose_jt[1,:] + np.array([ 0.00000000e+00 , 1.38777878e-17 ,-4.67753410e-03])
        agym_jt_smpl[18,:] = smpl_pose_jt[8,:] + np.array([ 0.00000000e+00 , 3.72529049e-09 ,-1.32426843e-02])
        agym_jt_smpl[19,:] = smpl_pose_jt[7,:] + np.array([ 0.00000000e+00 , 1.94289029e-16 ,-4.67755646e-03])

        agym_jt_smpl = agym_jt_smpl-agym_jt_smpl[3]+np.array(self.human_pos_offset)+np.array([0.0,0.0,+0.3]) #base pose
        
        return  agym_jt_smpl
