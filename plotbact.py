import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

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


def plot_reversals(particle_parameters):
    #To quantify reversal

    df = particle_parameters

    # Calculate the sign of velocities
    df['vx_sign'] = df['vx [um/s]'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['vy_sign'] = df['vy [um/s]'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    def count_reversals(group):
        group['vx_sign_change'] = group['vx_sign'].diff() != 0
        group['vy_sign_change'] = group['vy_sign'].diff() != 0
        group['reversal'] = (group['vx_sign_change'] & group['vy_sign_change']).astype(int)
        return group['reversal'].sum()

    # Apply the function to each group and get the count of reversals per particle
    reversals_per_particle = df.groupby('particle').apply(count_reversals)

    reversals_df = reversals_per_particle.reset_index()
    reversals_df.columns = ['Particle', 'Reversals']
    reversals_df.index.name = 'Index'

    average_reversals = reversals_df['Reversals'].mean()
    std_dev_reversals = reversals_df['Reversals'].std()

    print("Indexation of Groups (particles) and their reversals:")
    print(reversals_df)
    print(f"\nTotal number of unique particles: {len(reversals_df)}")
    print(f"\nAverage number of reversals: {average_reversals}")
    print(f"Standard deviation of reversals: {std_dev_reversals}")

    plt.figure(figsize=(10, 6))
    plt.hist(reversals_df['Reversals'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(average_reversals, color='red', linestyle='dashed', linewidth=2, label='Average reversals')
    plt.title('Histogram of reversal counts per particle')
    plt.xlabel('Number of reversals [-]')
    plt.ylabel('Number of bacteria [-]')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mean_velocities(particle_parameters):
    #To calculate mean velocties
    df=particle_parameters
    mean_velocities_per_particle = df.groupby('particle')[['vx [um/s]', 'vy [um/s]', 'vel [um/s]']].mean()
    std_dev_velocities_per_particle = df.groupby('particle')[['vx [um/s]', 'vy [um/s]', 'vel [um/s]']].std()

    print(f"Mean velocities for each particle:\n {mean_velocities_per_particle}")
    print(f"Standard deviations for each particle:\n {std_dev_velocities_per_particle}")

    overall_mean_velocity = mean_velocities_per_particle.mean()
    overall_std_dev_velocity = mean_velocities_per_particle.std()
    print(f"Overall mean velocity for all particles:\n {overall_mean_velocity}")
    print(f"Overall standard deviation for all particles:\n {overall_std_dev_velocity}")


    plt.figure(figsize=(10, 6))
    plt.hist(mean_velocities_per_particle['vx [um/s]'], bins=20, alpha=0.5, label='vx [um/s]', edgecolor='black')
    plt.hist(mean_velocities_per_particle['vy [um/s]'], bins=20, alpha=0.5, label='vy [um/s]', edgecolor='black')
    plt.hist(mean_velocities_per_particle['vel [um/s]'], bins=20, alpha=0.5, label='vel [um/s]', edgecolor='black')
    plt.axvline(overall_mean_velocity['vx [um/s]'], color='blue', linestyle='dashed', linewidth=2, label='Mean vx')
    plt.axvline(overall_mean_velocity['vy [um/s]'], color='orange', linestyle='dashed', linewidth=2, label='Mean vy')
    plt.axvline(overall_mean_velocity['vel [um/s]'], color='green', linestyle='dashed', linewidth=2, label='Mean vel')
    plt.title('Histogram of mean velocities')
    plt.xlabel('Mean velocity [um/s]')
    plt.ylabel('Number of bacteria [-]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_traj_length(particle_parameters):
    #To get total legths of the trajectories
    df=particle_parameters
    df = df.sort_values(by=['particle', 'frame'])

    df['x_shifted'] = df.groupby('particle')['x [um]'].shift(1)
    df['y_shifted'] = df.groupby('particle')['y [um]'].shift(1)
    df['distance'] = np.sqrt((df['x [um]'] - df['x_shifted'])**2 + (df['y [um]'] - df['y_shifted'])**2)

    total_distance_per_particle = df.groupby('particle')['distance'].sum()

    mean_trajectory_length = total_distance_per_particle.mean()
    std_trajectory_length = total_distance_per_particle.std()

    print(f"Total distance for each particle:\n {total_distance_per_particle}")
    print(f"Mean trajectory length: {mean_trajectory_length}")
    print(f"Standard deviation of trajectory length: {std_trajectory_length}")

    plt.figure(figsize=(10, 6))
    plt.hist(total_distance_per_particle, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(mean_trajectory_length, color='red', linestyle='dashed', linewidth=2, label='Mean trajectory length')
    plt.title('Histogram of trajectory lengths')
    plt.xlabel('Trajectory length [um]')
    plt.ylabel('Number of bacteria [-]')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_traj_end_to_end_lentgth(particle_parameters):
    #To get end-to-end distances of the trajectories
    df=particle_parameters
    grouped = df.groupby('particle')

    # Calculate the end-to-end distance for each particle
    total_end_to_end_distance_per_particle = grouped.apply(lambda x: np.sqrt((x['x [um]'].iloc[-1] - x['x [um]'].iloc[0])**2 + 
                                                (x['y [um]'].iloc[-1] - x['y [um]'].iloc[0])**2))

    # Calculate mean and standard deviation
    mean_distance = total_end_to_end_distance_per_particle.mean()
    std_distance = total_end_to_end_distance_per_particle.std()
    print(f"End-to-end trajectory length for each particle:\n {total_end_to_end_distance_per_particle}")
    print(f"Mean end-to-end trajectory length: {mean_distance}")
    print(f"Standard deviation of end-to-end trajectory length: {std_distance}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(total_end_to_end_distance_per_particle, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=2, label='Mean end-to-end trajectory length')
    plt.title('Histogram of end-to-end trajectory lengths')
    plt.xlabel('End-to-end trajectory length [um]')
    plt.ylabel('Number of bacteria [-]')
    plt.legend()
    plt.grid(True)
    plt.show()        