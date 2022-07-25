import numpy as np
import smplx
import torch
import pickle
import trimesh
from .smpl_dict import SMPLDict
from pytorch3d import transforms
import os

smplx_path = os.path.expanduser("~") +  "/hrl_git/assistive_gym_fem_tri/assistive_gym/envs/assets/smpl_models"
example_pkl_path = os.path.expanduser("~") + "/hrl_git/assistive_gym_fem_tri/assistive_gym/SAMP/pkl/armchair001_stageII.pkl"

model_path = "models"
gender = "male"

joint_dict = SMPLDict().joint_dict


def translate_mesh(mesh, xyz_tuple):
    n_points, _ = mesh.vertices.shape
    for i in range(n_points):
        mesh.vertices[i, 0] += xyz_tuple[0]
        mesh.vertices[i, 1] += xyz_tuple[1]
        mesh.vertices[i, 2] += xyz_tuple[2]

    return mesh


def show_smplx(body_model, model_output):
    vertices, faces, joints = extract_model_output(body_model, model_output)
    # trimesh config
    face_colors = np.ones([body_model.faces.shape[0], 4]) * [0.7, 0.7, 0.7, 0.5]

    # generate trimesh
    m = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

    # create scene
    scene = m.scene()

    # plot joints
    for entry in joint_dict:
        joint_ball = trimesh.creation.uv_sphere(radius=0.01)
        joint_ball.visual.vertex_colors = [1., 0., 0., 1.]
        joint_ball = translate_mesh(joint_ball, joints[joint_dict[entry]])
        scene.add_geometry(joint_ball)

    scene.show()


def extract_model_output(body_model, model_output):
    # extract mesh
    vertices = model_output.vertices.detach().numpy().squeeze()
    faces = body_model.faces
    joints = model_output.joints.detach().numpy().squeeze()

    return vertices, faces, joints


def load_pose(pose):
    body_model = smplx.create(model_path=smplx_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=1)

    # extract default pkl stuff (just for shape)
    with open(example_pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1, 10)

    translation = torch.tensor([0., 0., 0.]).reshape(1, -1)
    orientation = pose[0:3].reshape(1, -1)

    # reshape pose to sort for each joint
    reshaped_pose = pose.reshape(55, 3)

    input_pose = pose[3:66].reshape(1, -1)

    # run model
    model_output = body_model(global_orient=orientation, body_pose=input_pose, betas=betas, transl=translation,
                              return_verts=True)

    return body_model, model_output, reshaped_pose, orientation, translation


def load_pose_from_pkl(pkl_path, pose_number):
    # create body model
    body_model = smplx.create(model_path=smplx_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=1)

    # extract pkl stuff
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1, 10)
        full_trans = torch.tensor(data['pose_est_trans'], dtype=torch.float32)

    full_pose = full_poses[pose_number, :]
    translation = full_trans[pose_number, :].reshape(1, -1)

    orientation = full_pose[0:3].reshape(1, -1)
    pose = full_pose[3:66].reshape(1, -1)

    # reshape pose to sort for each joint
    reshaped_pose = full_pose.reshape(55, 3)

    # Run model
    model_output = body_model(global_orient=orientation, body_pose=pose, betas=betas, transl=translation,
                              return_verts=True)

    return body_model, model_output, reshaped_pose, orientation, translation