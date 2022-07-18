import os, sys, copy, glob, json, time, random, argparse
from termios import B1000000
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo
from lib.load_data import load_data
from lib.load_blender import pose_spherical

import pickle
from geomloss import SamplesLoss # type: ignore



def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_fine_only", type=str, default='')
    parser.add_argument("--morph", type=str, default=None)
    parser.add_argument("--baseline", action='store_true')


    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_sphere", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    # morph
    parser.add_argument("--input", action='store_true')

    return parser

def padding(alpha, wld_size, val=0):
    wld_size = wld_size.cpu().numpy()
    sh = torch.tensor(alpha.size()).cpu().numpy()
    sh = (wld_size - sh)/2
    sh_flr = np.floor(sh).astype(np.int32)
    sh_ceil = np.ceil(sh).astype(np.int32)
    pad = (sh_flr[2],sh_ceil[2],sh_flr[1],sh_ceil[1],sh_flr[0],sh_ceil[0])
    a = F.pad(alpha,pad,value=val)
    return a

def vlm2cld_rgba(alpha, rgb, dtype): # normalized images (between 0 and 1)
    ind = alpha.nonzero()
    indx = ind[:,0]
    indy = ind[:,1]
    indz = ind[:,2]
    a_i = alpha[indx, indy, indz]
    alpha = alpha / alpha.sum()
    p_i = alpha[indx, indy, indz]
    rgb_i = rgb[indx, indy, indz, :]

    return ind.type(dtype), a_i, p_i, rgb_i

def color_correspondence(tgt,nx,rgb_y):
    tgt1 = tgt.round().type(torch.long)
    K=1
    tgt1[tgt1<K] = K
    tgt1[tgt1>(nx-(K+1))] = nx-(K+1)
    rb1 = []
    for i in range(-K,K+1):
        for j in range(-K,K+1):
            for k in range(-K,K+1):
                rb1.append(rgb_y[tgt1[:,0]+i,tgt1[:,1]+j,tgt1[:,2]+k])

    rb1 = torch.stack(rb1,0)
    rb_mean = rb1.mean(0)
    return rb_mean

def cld2vlm(pc, nx, ny, nz, weights=None):
    x,y,z = pc[:,0], pc[:,1], pc[:,2]
    x0, x1, xa, xb = x.floor(), x.ceil(), x.frac(), (1 - x.frac())
    y0, y1, ya, yb = y.floor(), y.ceil(), y.frac(), (1 - y.frac())
    z0, z1, za, zb = z.floor(), z.ceil(), z.frac(), (1 - z.frac())
    N = pc.shape[0]
    cld = pc.repeat(8,1)
    cld[N*0:N*1,0], cld[N*0:N*1,1], cld[N*0:N*1,2] = x0, y0, z0
    cld[N*1:N*2,0], cld[N*1:N*2,1], cld[N*1:N*2,2] = x0, y0, z1
    cld[N*2:N*3,0], cld[N*2:N*3,1], cld[N*2:N*3,2] = x0, y1, z0
    cld[N*3:N*4,0], cld[N*3:N*4,1], cld[N*3:N*4,2] = x0, y1, z1
    cld[N*4:N*5,0], cld[N*4:N*5,1], cld[N*4:N*5,2] = x1, y0, z0
    cld[N*5:N*6,0], cld[N*5:N*6,1], cld[N*5:N*6,2] = x1, y0, z1
    cld[N*6:N*7,0], cld[N*6:N*7,1], cld[N*6:N*7,2] = x1, y1, z0
    cld[N*7:N*8,0], cld[N*7:N*8,1], cld[N*7:N*8,2] = x1, y1, z1
    oct_weights = weights.repeat(8)
    oct_weights[N*0:N*1] = weights * xb * yb * zb
    oct_weights[N*1:N*2] = weights * xb * yb * za
    oct_weights[N*2:N*3] = weights * xb * ya * zb
    oct_weights[N*3:N*4] = weights * xb * ya * za
    oct_weights[N*4:N*5] = weights * xa * yb * zb
    oct_weights[N*5:N*6] = weights * xa * yb * za
    oct_weights[N*6:N*7] = weights * xa * ya * zb
    oct_weights[N*7:N*8] = weights * xa * ya * za

    bins = (cld[:, 2]).round() + nz * (cld[:, 1]).round() + nz * ny * (cld[:, 0]).round()
    count = bins.int().bincount(weights=oct_weights, minlength=nx * ny * nz)
    return count.view(nx, ny, nz)

def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t+t0), radius * np.sin(r * t+t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t+t0), h, radius * np.sin(r * t+t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t+t0), radius * np.sin(r * t+t0)]


def generate_poses(t, speed=None,at=None,up=None,radius=None,axis=None, inv_RT=None, action='none'):
    path_gen = circle(radius=radius,axis=axis)
    if inv_RT is None:
        cam_pos = torch.tensor(path_gen(t * speed / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos,at=at, up=up, inverse=True, cv=True)
        
        inv_RT = cam_pos.new_zeros(4, 4)
        inv_RT[:3, :3] = cam_rot
        inv_RT[:3, 3] = cam_pos
        inv_RT[3, 3] = 1
    else:
        inv_RT = torch.from_numpy(inv_RT)
    
    return inv_RT

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """
       
    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2==0] = 1
        return x / l2, l2

def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)

def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)

def morphing(args,cfg):

    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # load data
    A,B,w = args.morph.split(',')
    with open(f'morph_data/{A}.pkl', "rb") as ff:
        data_A = pickle.load(ff)
    with open(f'morph_data/{B}.pkl', "rb") as ff:
        data_B = pickle.load(ff)
    w = float(w)
    nerf = True
    mixed = False
    if nerf:
        radius = 4
        N_frame = 6
        render_poses = torch.stack([pose_spherical(angle, -30.0, radius) for angle in np.linspace(-180,-90,N_frame+1)[:-1]], 0)
        
    elif mixed:
        at = torch.tensor([0.,0.5,0.]) 
        up = [0,1,0]
        render_poses = [generate_poses(t,speed=30,at=at,up=up,axis='y',radius=2) for t in range(12)]
        render_poses = torch.stack(render_poses,0).type('torch.cuda.FloatTensor')
    else:
        at = torch.tensor([0.,0.,0.2]) 
        up = [0,0,-1]
        render_poses = [generate_poses(t,speed=30,at=at,up=up,axis='z',radius=2) for t in range(12)]
        render_poses = torch.stack(render_poses,0).type('torch.cuda.FloatTensor')

    # take largest world size
    xyz_min = torch.minimum(data_A['xyz_min'], data_B['xyz_min'])
    xyz_max = torch.maximum(data_A['xyz_max'], data_B['xyz_max'])
    xyz_min = xyz_min.min().repeat(3)
    xyz_max = xyz_max.max().repeat(3)
    # align
    alpha_x = data_A['alpha'].squeeze()
    alpha_y = data_B['alpha'].squeeze()
    ratio_A = torch.Tensor([alpha_x.shape])/(data_A['xyz_max']-data_A['xyz_min'])
    ratio_B = torch.Tensor([alpha_y.shape])/(data_B['xyz_max']-data_B['xyz_min'])
    ptspermeter = 100
    ratio_A = ptspermeter / ratio_A.squeeze()
    ratio_B = ptspermeter / ratio_B.squeeze()
    # alpha
    alpha_x_ori = F.interpolate(data_A['alpha'],scale_factor=(ratio_A[0].item(),ratio_A[1].item(),ratio_A[2].item()),mode='trilinear').squeeze()
    alpha_y_ori = F.interpolate(data_B['alpha'],scale_factor=(ratio_B[0].item(),ratio_B[1].item(),ratio_B[2].item()),mode='trilinear').squeeze()
    voxel_size_d = 1 / ptspermeter
    world_size_d = ((xyz_max - xyz_min) / voxel_size_d).long()
    nx, ny, nz = world_size_d
    alpha_x = padding(alpha_x_ori, world_size_d)
    alpha_y = padding(alpha_y_ori, world_size_d)
    # rgb
    rgb_x_ori = F.interpolate(data_A['rgb'],scale_factor=(ratio_A[0].item(),ratio_A[1].item(),ratio_A[2].item()),mode='trilinear').squeeze().permute(1,2,3,0)
    rgb_y_ori = F.interpolate(data_B['rgb'],scale_factor=(ratio_B[0].item(),ratio_B[1].item(),ratio_B[2].item()),mode='trilinear').squeeze().permute(1,2,3,0)
    rgb_x = []
    rgb_y = []
    for i in range(3):
        rgb_x.append(padding(rgb_x_ori[...,i], world_size_d, val=0))
        rgb_y.append(padding(rgb_y_ori[...,i], world_size_d, val=0))
    rgb_x = torch.stack(rgb_x, -1)
    rgb_y = torch.stack(rgb_y, -1)
    
    
    # load model
    ckpt_path = os.path.join(cfg.basedir, f'dvgo_{A}', 'fine_last.tar')
    model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
    stepsize = cfg.fine_model_and_render.stepsize
    render_kwargs = {
        'near': data_A['near'],
        'far': data_A['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    # import pdb; pdb.set_trace()
    # modify model
    model.xyz_min = xyz_min
    model.xyz_max = xyz_max
    model.world_size = world_size_d
    model.voxel_size = voxel_size_d
    model.voxel_size_ratio = model.voxel_size / model.voxel_size_base

    ### OT ###
    # threshold to filter small weight
    rgb_x[alpha_x < 1e-3,:] = 0
    rgb_y[alpha_y < 1e-3,:] = 0
    thres = 0.2
    alpha_x[alpha_x<thres] = 0
    alpha_y[alpha_y<thres] = 0
    print('point number:', alpha_x.count_nonzero(), alpha_y.count_nonzero())

    # color distributoin: mean, std
    srA,rA = torch.std_mean(rgb_x[alpha_x>0][:,0])
    sgA,gA = torch.std_mean(rgb_x[alpha_x>0][:,1])
    sbA,bA = torch.std_mean(rgb_x[alpha_x>0][:,2])
    srB,rB = torch.std_mean(rgb_y[alpha_y>0][:,0])
    sgB,gB = torch.std_mean(rgb_y[alpha_y>0][:,1])
    sbB,bB = torch.std_mean(rgb_y[alpha_y>0][:,2])
    print('A,',rA,gA,bA,srA,sgA,sbA)
    print('B,',rB,gB,bB,srB,sgB,sbB)

    # volume to point cloud
    aa, ai, pa, ra = vlm2cld_rgba(alpha_x,rgb_x,dtype)
    ab, bi, pb, rb = vlm2cld_rgba(alpha_y,rgb_y,dtype)
    aa = aa.contiguous()
    ab = ab.contiguous()
   
    rigid_transform = True
    barycenter = True

    # 6DoF
    if rigid_transform:
        aa = aa / nx
        ab = ab / nx
        R = torch.eye(3).type(dtype)
        R.requires_grad = True
        T = torch.zeros(3).type(dtype)
        T.requires_grad = True
        loss = SamplesLoss("sinkhorn", p=2, blur=0.1)
        Nsteps = 100
        Nshow = 10
        lr = 1
        for i in range(Nsteps):
            aa_local = aa @ R.T + T
            L_ab = loss(aa_local, ab)
            [g_R, g_T] = torch.autograd.grad(L_ab, [R,T], create_graph=True)
            R_lcl = R - lr * g_R
            u,s,vh = torch.linalg.svd(R_lcl)
            if (i+1) % Nshow == 0:
                print(i,L_ab)
            T.data = T - lr * g_T
            R.data = u @ vh
        print(R)
        print(T)
        T = T * nx
        aa = aa * nx
        ab = ab * nx

    # Barycenter #####
    if barycenter:
        loss_bary = SamplesLoss("sinkhorn", p=2, blur=1, scaling=0.9)
        az = aa.clone()
        az.requires_grad = True
        if rigid_transform:
            az = aa @ R.T + T
        pz = pa.clone()
        
        data = [ab]#[aa,ab]
        dist = [pb]#[pa,pb]
        margin = []
        margin.append(aa)
        for i in range(1):
            L_ab = loss_bary(pz, az, dist[i], data[i])
            print(i,L_ab)
            [g] = torch.autograd.grad(L_ab, [az])
            margin.append(az - g / pz.view(-1, 1))

        rb = color_correspondence(margin[1],nx,rgb_y)
        print('rb_corr,',rb.mean(0))

    ### render ###
    # morph
    aa_w = (1-w) * margin[0] + w * margin[1]
    rw = (1-w) * ra + w * rb
    aa_w[aa_w<0] = 0.
    aa_w[aa_w>(nx-1)] = nx-1.
    alpha = cld2vlm(aa_w, nx, ny, nz, ai)
    alpha = alpha / alpha.max()
    rgb = torch.zeros_like(rgb_x)
    rgb[...,0] = cld2vlm(aa_w, nx, ny, nz, rw[:,0])
    rgb[...,1] = cld2vlm(aa_w, nx, ny, nz, rw[:,1])
    rgb[...,2] = cld2vlm(aa_w, nx, ny, nz, rw[:,2])
    # gray world
    srM,rM = torch.std_mean(rgb[alpha>0][:,0])
    sgM,gM = torch.std_mean(rgb[alpha>0][:,1])
    sbM,bM = torch.std_mean(rgb[alpha>0][:,2])
    srW,rW = ((1-w) * srA + w * srB),((1-w) * rA + w * rB)
    sgW,gW = ((1-w) * sgA + w * sgB),((1-w) * gA + w * gB)
    sbW,bW = ((1-w) * sbA + w * sbB),((1-w) * bA + w * bB)
    color_shift = True
    if color_shift:
        gain = 1.0
        offset = -0.05
        rgb[...,0] = gain* ((rgb[...,0] - rM) / srM * srW + rW) + offset
        rgb[...,1] = gain* ((rgb[...,1] - gM) / sgM * sgW + gW) + offset
        rgb[...,2] = gain* ((rgb[...,2] - bM) / sbM * sbW + bW) + offset        
    print('size,',(ab.shape[0]/aa.shape[0]))

    alpha_M = alpha[None, None, ...].type(dtype)
    rgb_M = rgb.permute(3,0,1,2)[None,...]
    render_alpha = alpha_M
    render_rgb = rgb_M


    bound = False
    if bound:
        alpha = render_alpha.squeeze()
        rgb = render_rgb.squeeze().permute(1,2,3,0)
        alpha[:,0,0] = 1; alpha[:,-1,0] = 1; alpha[:,0,-1] = 1; alpha[:,-1,-1] = 1
        alpha[0,:,0] = 1; alpha[-1,:,0] = 1; alpha[0,:,-1] = 1; alpha[-1,:,-1] = 1
        alpha[0,0,:] = 1; alpha[-1,0,:] = 1; alpha[0,-1,:] = 1; alpha[-1,-1,:] = 1
        rgb[:,0,0,0] = 1; rgb[:,-1,0,0] = 1; rgb[:,0,-1,0] = 1; rgb[:,-1,-1,0] = 1
        rgb[0,:,0,0] = 1; rgb[-1,:,0,0] = 1; rgb[0,:,-1,0] = 1; rgb[-1,:,-1,0] = 1
        rgb[0,0,:,0] = 1; rgb[-1,0,:,0] = 1; rgb[0,-1,:,0] = 1; rgb[-1,-1,:,0] = 1
        render_alpha = alpha[None, None, ...]
        render_rgb = rgb.permute(3,0,1,2)[None,...]

    rgbs = []
    disps = []
    HW = data_A['HW']
    Ks = data_A['Ks']

    render_factor = 2
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor
    
    ckpt_name = f'{A}_{B}_{w}'
    
    savedir = os.path.join(cfg.basedir, cfg.expname, f'debias_{ckpt_name}')
    os.makedirs(savedir, exist_ok=True)

    # render_poses = render_poses[:10]
    print('poses,',render_poses.size())
    with torch.no_grad():
        for i, c2w in enumerate(tqdm(render_poses)):
            N_rows = 50
            H, W = HW[i]
            K = Ks[i]
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc=False, inverse_y=render_kwargs['inverse_y'],
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            keys = ['rgb_marched', 'disp']
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, render_alpha=render_alpha, render_rgb=render_rgb, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(N_rows, 0), rays_d.split(N_rows, 0), viewdirs.split(N_rows, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks])
                for k in render_result_chunks[0].keys()
            }
            rgb = render_result['rgb_marched'].cpu().numpy()
            disp = render_result['disp'].cpu().numpy()

            rgbs.append(rgb)
            disps.append(disp)
            if i==0:
                print('Testing', rgb.shape, disp.shape)
            if savedir is not None:
                rgb8 = utils.to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
                if bound:
                    rgb8 = utils.to8b(disps[-1] / np.max(disps[-1]))
                    filename = os.path.join(savedir, '{:03d}d.png'.format(i))
                    imageio.imwrite(filename, rgb8)

        rgbs = np.array(rgbs)
        disps = np.array(disps)
        imageio.mimwrite(os.path.join(savedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(savedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)   
        print(f'{A}_{B}_{w},done!')
    return



@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs, render_alpha=None, render_rgb=None,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    render_factor=2
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    bound = False
    alpha = None
    render_rgb = None
    if bound:
        alpha = model.activate_density(model.density).squeeze()
        rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0)
        alpha[:,0,0] = 1; alpha[:,-1,0] = 1; alpha[:,0,-1] = 1; alpha[:,-1,-1] = 1
        alpha[0,:,0] = 1; alpha[-1,:,0] = 1; alpha[0,:,-1] = 1; alpha[-1,:,-1] = 1
        alpha[0,0,:] = 1; alpha[-1,0,:] = 1; alpha[0,-1,:] = 1; alpha[-1,-1,:] = 1
        rgb[:,0,0,0] = 1; rgb[:,-1,0,0] = 1; rgb[:,0,-1,0] = 1; rgb[:,-1,-1,0] = 1
        rgb[0,:,0,0] = 1; rgb[-1,:,0,0] = 1; rgb[0,:,-1,0] = 1; rgb[-1,:,-1,0] = 1
        rgb[0,0,:,0] = 1; rgb[-1,0,:,0] = 1; rgb[0,-1,:,0] = 1; rgb[-1,-1,:,0] = 1
        render_alpha = alpha[None, None, ...]
        render_rgb = rgb.permute(3,0,1,2)[None,...]

    rgbs = []
    disps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):
        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, render_alpha=render_alpha, render_rgb=render_rgb, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        rgbs.append(rgb)
        disps.append(disp)
        if i==0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            if bound:
                rgb8 = utils.to8b(disps[-1] / np.max(disps[-1]))
                filename = os.path.join(savedir, '{:03d}d.png'.format(i))
                imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        '''
        print('Testing psnr', [f'{p:.3f}' for p in psnrs])
        if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
        if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
        if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
        '''
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    return rgbs, disps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model = dvgo.DirectVoxGO(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            model.scale_volume_grid(model.num_voxels * 2)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_tv_density>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_density * model.density_total_variation()
        if cfg_train.weight_tv_k0>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_k0 * model.k0_total_variation()
        loss.backward()
        optimizer.step()
        psnr_lst.append(psnr)

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'MaskCache_kwargs': model.get_MaskCache_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
            data_dict=data_dict, stage='coarse')
    eps_coarse = time.time() - eps_coarse
    eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
    print('train: coarse geometry searching in', eps_time_str)

    # fine detail reconstruction
    eps_fine = time.time()
    coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
            model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
            thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    if args.morph is not None:
        morphing(args, cfg)
        print('done')
        sys.exit()

    ### train ###
    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)
    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    if args.export_fine_only:
        print('Export fine visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density)
            rgb = torch.sigmoid(model.k0)
            xyz_min=model.xyz_min
            xyz_max=model.xyz_max
            data = dict(
                alpha=alpha, rgb=rgb,
                xyz_min=xyz_min, xyz_max=xyz_max,
                HW=data_dict['HW'], Ks=data_dict['Ks'],
                near=data_dict['near'], far=data_dict['far'],
                poses=data_dict['render_poses'],
            )
            with open(f'morph_data/{args.export_fine_only}.pkl', "wb") as ff:
                pickle.dump(data, ff)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.render_sphere:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_sphere_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render sphere
    nerf = True
    if nerf:
        radius = 4
        phi = -30
        N_frame = 8
        render_poses = torch.stack([pose_spherical(theta, phi, radius) for theta in np.linspace(-180,180,N_frame+1)[:-1]], 0)
        # render_poses[:,:,1] *= -1
        # render_poses[:,:,2] *= -1
    else:
        # at = torch.tensor([0.5,0.5,0.5])
        at = torch.tensor([0,0,0.2])
        up = [0,0,-1]
        render_poses = [generate_poses(t,speed=30,at=at,up=up,axis='z',radius=2) for t in range(12)]
        render_poses = torch.stack(render_poses,0).type('torch.cuda.FloatTensor')

    if args.render_sphere:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_sphere_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
                render_poses=render_poses,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    print('Done')


