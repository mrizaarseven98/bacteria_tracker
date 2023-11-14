import numpy as np
import cv2
from PIL import Image
import pandas as pd
from skimage.io import imread,imshow
import matplotlib.pyplot as plt
from tqdm import tqdm
import trackpy as tp
import seaborn
import os


def load_tif(filename, show=True):

    # read image using your favorite package
    tif_file = imread(filename)
    im=tif_file[0,:,:]

    if show:
        im_inverted=np.invert(im)
        imshow(im_inverted)
        plt.title("First segmented frame of the video")
        plt.show()
    return tif_file

def contour_fitting(tif_file, show_nth_frame=None):

    tif_dimension=tif_file.ndim
    
    if tif_dimension==3:
        if show_nth_frame is not None:
            nthframe=tif_file[show_nth_frame,:,:]
            img_where=np.where(nthframe==0)
            blank_image = np.zeros((nthframe.shape[0],nthframe.shape[1],3), np.uint8)
                    
            for i in range(0, len(img_where[0])):
                blank_image[img_where[0][i]][img_where[1][i]]=(255,255,255)
                
        cnt_image=blank_image

        contours_list=[]
        for nframe in tqdm(range(0, tif_file.shape[0]), desc="Finding the contours of the bacteria..."):
            frame=tif_file[nframe]
            frame=frame.astype(np.uint8)
            
            #print(np.shape(blank_image))

            #Detecting contours of the bacteria
            contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_list.append(contours)

            #Drawing the contours in red
            if show_nth_frame is not None and nframe==show_nth_frame:

                cv2.drawContours(cnt_image, contours, -1, (255,0,0), 1)

        if show_nth_frame is not None:
                plt.imshow(cnt_image)
                plt.title("Bacterial contours drawn on the " + str(show_nth_frame)+"th image")
                plt.show()
        #print(len(contours_list))
        return contours_list
    else:
        raise ValueError("Shape of the tif file should be (# of frames, dim_x, dim_y)")
        
    
def fit_ellipses(tif_file, contours=None, filter_max=None, show_nth_frame=None, debug=False):
    
    if contours is None:
        contours=contour_fitting(tif_file, show_nth_frame=show_nth_frame)

    if show_nth_frame is not None:
        nthframe=tif_file[show_nth_frame,:,:]
        img_where=np.where(nthframe==0)
        blank_image = np.zeros((nthframe.shape[0],nthframe.shape[1],3), np.uint8)
                
        for i in range(0, len(img_where[0])):
            blank_image[img_where[0][i]][img_where[1][i]]=(255,255,255)
                
        ellipse_image=blank_image

    tif_dimension=tif_file.ndim
    
    if tif_dimension==3:
        
        ellipses_all_frames=[]
        for nframe in tqdm(range(0, tif_file.shape[0]), "Fitting ellipses to the contours..."):
            mistreated_ellipses=0
            ellipses=[]

            for j in range(0, len(contours[nframe])):   
                
                try:
                    #Fitting an elipse to the bacteria contours
                    ellipse = cv2.fitEllipse(contours[nframe][j])
                    if filter_max is not None and not(1<ellipse[1][0]<filter_max[0] and 1<ellipse[1][1]<filter_max[1]):
                        raise Exception("Filtering...")
                    else:
                        ellipses.append(ellipse)

                except Exception as e:
                    if(debug):
                        print(e)
                        mistreated_ellipses+=1
            
                if nframe==show_nth_frame and len(ellipses)!=0:
                    cv2.ellipse(ellipse_image,ellipse, (0,0,255), 1)

            if debug:
                print("Number of undetected ellipses: "+ str(mistreated_ellipses)+ " , in frame: "+str(nframe))

            ellipses_all_frames.append(ellipses)
        if show_nth_frame is not None:
                plt.imshow(ellipse_image)
        return ellipses_all_frames
    else:
        raise ValueError("Shape of the tif file should be (# of frames, dim_x, dim_y)")
    
def format_ellipses(ellipses_df):
    all_ellipses = [(frame_idx, ellipse) for frame_idx, frame in enumerate(ellipses_df) for ellipse in frame]
    
    #We then put it in a pandas dataframe
    ellipses_df = pd.DataFrame({
        'x': [ellipse[0][0] for _, ellipse in all_ellipses],
        'y': [ellipse[0][1] for _, ellipse in all_ellipses],
        'frame': [frame_idx for frame_idx, _ in all_ellipses],
        'major_axis_size': [ellipse[1][1] for _, ellipse in all_ellipses], #length of major axis
        'minor_axis_size': [ellipse[1][0] for _, ellipse in all_ellipses], #length of major axis
        'body_angle': [ellipse[2] for _, ellipse in all_ellipses], #alignment angle of the ellipse
        
    })

    ellipses_df=ellipses_df.dropna()
    return ellipses_df

def ellipse_to_particle(tif_file, file_name, ellipses_df):

    output_path=file_name[:-4]+str("_particle.tif")
    image_to_save=[]
    ellipses_group=ellipses_df.groupby("frame")
    
    for nframe, group in tqdm(ellipses_group, "Converting ellipses to particles"):
        blank_image = np.zeros((tif_file.shape[1],tif_file.shape[2],3), np.uint8)
        
        for index, row in group.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            cv2.circle(blank_image, (x,y), radius=1, color=(255, 255, 255), thickness=-1)

        image_to_save.append(Image.fromarray(blank_image))

    image_to_save[0].save(output_path, save_all=True, append_images=image_to_save[1:], compression='tiff_deflate')
    

def track_bacteria(ellipses_df, max_search_range=20, min_search_range=10, filter=None):
    traj_tp = tp.link_df(ellipses_df, 
                  search_range = max_search_range, 
                  adaptive_stop = min_search_range,
                  adaptive_step=0.98, 
                  memory = 0)
    traj_tp = traj_tp.sort_values(by=['particle', 'frame'])

    traj_tp.reset_index(drop=True, inplace=True)
    
    if filter is not None:
        traj_tp=filter_trajectories(traj_tp,filter_size=filter)

    return traj_tp
        
def filter_trajectories(trajectory, filter_size=100):
    
    particle_counts = trajectory['particle'].value_counts()
    particles_to_keep = particle_counts[particle_counts > filter_size].reset_index().rename(columns={'index': 'particle', 'particle': 'count'})
    display(particles_to_keep)
    trajectory_filtered = pd.merge(trajectory, particles_to_keep[['particle']], on='particle')

    return trajectory_filtered

def write_trajectory(trajectory, filename):

    dir_name = "trajectory_data"
    full_dir_path = os.path.join(dir_name, filename)
    
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)

    pickle_file_path = os.path.join(full_dir_path, "trajectory.csv")
    trajectory.to_csv(pickle_file_path, index=False)
    
def write_kinematics(kinematics_data, filename):
    
    dir_name = "trajectory_data"
    full_dir_path = os.path.join(dir_name, filename)

    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)

    kinematics_file_path = os.path.join(full_dir_path, "kinematics.csv")
    kinematics_data.to_csv(kinematics_file_path, index=False)
    

def read_trajectory(filename):

    dir_name = "trajectory_data"
    full_dir_path = os.path.join(dir_name, filename)

    if os.path.exists(full_dir_path):

        csv_file_path = os.path.join(full_dir_path, "trajectory.csv")
        trajectory=pd.read_csv(csv_file_path)

        return trajectory
    
    else:
        raise OSError

#We take this function from a python package called pytaxis: https://github.com/tatyana-perlova/pytaxis
def plot_traj_all(traj,
                imdim1,
                imdim2,
                pix_size, 
                palette = None, 
                scalebar = 100):

    if palette == None:
        palette = seaborn.color_palette("Dark2", len(traj.particle.unique()))
    plt.tick_params(\
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',
        left = 'off',
        right = 'off',# ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.yticks([])
    plt.xticks([])
    plt.xlim(0, imdim1)
    plt.ylim(0, imdim2)
    
    
    unstacked = traj.set_index(['frame', 'particle']).unstack()
    plot = plt.plot(unstacked.x, unstacked.y, linewidth=2, alpha = 1)
    plt.gca().set_aspect(1)
    plt.gca().invert_yaxis()
    plt.plot([1550, 1550 + scalebar/pix_size], [350 , 350], color = 'black', linewidth = 4)
    plt.text(1550, 300, r'{}$\mu m$'.format(scalebar), fontsize = 18)

def calculate_kinematic_parameters(trajectory_data, time_step=1, pixel_size=0.1625):
    
    def group_parameters(traj_group):
        
        # If time_step is set to 1, it will calculate the displacement parameters of particles in 2 consecutive frames
        traj_group['x']=traj_group['x']*pixel_size
        traj_group['y']=traj_group['y']*pixel_size
        traj_group['major_axis_size']=traj_group['major_axis_size']*pixel_size
        traj_group['minor_axis_size']=traj_group['minor_axis_size']*pixel_size
        
        # Calculate velocities
        traj_group['vx'] = traj_group['x'].diff() / time_step
        traj_group['vy'] = traj_group['y'].diff() / time_step
        traj_group['vel']=np.sqrt(traj_group['vx']**2 + traj_group['vy']**2)

        # Calculate accelerations
        traj_group['ax'] = traj_group['vx'].diff() / time_step
        traj_group['ay'] = traj_group['vy'].diff() / time_step
        traj_group['acc']=np.sqrt(traj_group['ax']**2 + traj_group['ay']**2)

        # Calculate angular position
        traj_group['diff_body_angle'] = traj_group['body_angle'].diff()

        # Calculate angular velocity
        traj_group['angular_vel'] = traj_group['diff_body_angle'].diff() / time_step

        # Calculate angular acceleration
        traj_group['angular_acc'] = traj_group['angular_vel'].diff() / time_step

        return traj_group
    
    tqdm.pandas()
    traj_params = trajectory_data.groupby('particle', group_keys=True).progress_apply(group_parameters)
    traj_params.reset_index(drop=True, inplace=True)

    return traj_params