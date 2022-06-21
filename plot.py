import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
sns.set()




# load simulation results
def plot_EV():
    fig, axs = plt.subplots(4,3, figsize=(10,9.5), sharex=True)
    axs[0, 0].plot([10, 20, 50], [0.91, 2.36, 10.40], '-o')
    axs[0, 0].plot([10, 20, 50], [0.91, 2.84, 10.76], '-^')
    axs[0, 0].plot([10, 20, 50], [0.46, 2.12, 16.53], '-s')
    axs[0, 0].set_title('auSHDC')
    axs[0, 0].set_ylabel('ER2')
    axs[0, 1].plot([10, 20, 50], [0.70, 2.80, 23.30], '-o')
    axs[0, 1].plot([10, 20, 50], [0.70, 3.70, 25.80], '-^')
    axs[0, 1].plot([10, 20, 50], [0.46, 3.00, 35.70], '-s')
    axs[0, 1].set_title('SHD(0.3)')
    axs[0, 2].plot([10, 20, 50], [0.93, 0.94, 0.90], '-o')
    axs[0, 2].plot([10, 20, 50], [0.93, 0.93, 0.91], '-^')
    axs[0, 2].plot([10, 20, 50], [0.95, 0.97, 0.88], '-s')
    axs[0, 2].set_title('auPRC')
    axs[1, 0].plot([10, 20, 50], [2.81, 6.14,  31.00], '-o')
    axs[1, 0].plot([10, 20, 50], [2.83, 6.59, 33.85], '-^')
    axs[1, 0].plot([10, 20, 50], [1.93, 8.20, 46.11], '-s')
    axs[1, 0].set_ylabel('ER4')
    axs[1, 1].plot([10, 20, 50], [3.40, 8.50, 74.70], '-o')
    axs[1, 1].plot([10, 20, 50], [3.60, 9.40, 83.70], '-^')
    axs[1, 1].plot([10, 20, 50], [1.80, 15.20, 141.30], '-s')
    axs[1, 2].plot([10, 20, 50], [0.96, 0.95, 0.89], '-o')
    axs[1, 2].plot([10, 20, 50], [0.96, 0.95, 0.88], '-^')
    axs[1, 2].plot([10, 20, 50], [0.97, 0.95, 0.87], '-s')
    axs[2, 0].plot([10, 20, 50], [0.91, 2.40, 4.61], '-o')
    axs[2, 0].plot([10, 20, 50], [0.91, 2.25, 4.60], '-^')
    axs[2, 0].plot([10, 20, 50], [0.58, 2.37, 12.76], '-s')
    axs[2, 0].set_ylabel('SF2')
    axs[2, 1].plot([10, 20, 50], [1.20, 3.90, 6.20], '-o')
    axs[2, 1].plot([10, 20, 50], [1.20, 3.50, 6.90], '-^')
    axs[2, 1].plot([10, 20, 50], [0.20, 2.20, 28.70], '-s')
    axs[2, 2].plot([10, 20, 50], [0.92, 0.95, 0.97], '-o')
    axs[2, 2].plot([10, 20, 50], [0.92, 0.95, 0.97], '-^')
    axs[2, 2].plot([10, 20, 50], [0.97, 0.98, 0.97], '-s')
    axs[3, 0].plot([10, 20, 50], [1.57, 4.47, 19.57], '-o')
    axs[3, 0].plot([10, 20, 50], [1.82, 6.12, 18.79], '-^')
    axs[3, 0].plot([10, 20, 50], [0.79, 3.51, 18.14], '-s')
    axs[3, 0].set_ylabel('SF4')
    axs[3, 1].plot([10, 20, 50], [2.10, 6.40, 35.90], '-o')
    axs[3, 1].plot([10, 20, 50], [2.50, 9.70, 32.70], '-^')
    axs[3, 1].plot([10, 20, 50], [0.80, 7.10, 55.30], '-s')
    axs[3, 2].plot([10, 20, 50], [0.95, 0.95, 0.94], '-o')
    axs[3, 2].plot([10, 20, 50], [0.95, 0.93, 0.97], '-^')
    axs[3, 2].plot([10, 20, 50], [0.93, 0.93, 0.90], '-s')


    fig.legend(["Our method", "NOTEARS", "GOLEM-EV"], loc="lower center", ncol = 3)
    fig.tight_layout()
    plt.savefig('homo-EV.png', dpi=500)
    # plt.show()
    
def plot_NV():
    fig, axs = plt.subplots(4,3, figsize=(10,9.5), sharex=True)

    # ER2
    axs[0, 0].set_ylabel('ER2')
    ## auSHDC
    axs[0, 0].set_title('auSHDC')
    axs[0, 0].plot([10, 20, 50], [1.61, 1.92, 2.14], '-o', label = 'Our method')
    axs[0, 0].plot([10, 20, 50], [1.92, 3.02, 13.74], '-^', label = 'NOTEARS')
    axs[0, 0].plot([10, 20, 50], [2.14, 3.95, 10.25], '-s', label = 'GOLEM_NV')
    # axs[0, 0].set_xlim([9.5, 50.5])
    axs[0, 0].set_xticks([10, 20, 50])
    
    ## SHD
    axs[0, 1].set_title('SHD(0.3)')
    axs[0, 1].plot([10, 20, 50], [2.00, 4.40, 19.80], '-o', label = 'Our method')
    axs[0, 1].plot([10, 20, 50], [2.70, 4.40, 24.40], '-^', label = 'NOTEARS')
    axs[0, 1].plot([10, 20, 50], [3.20, 5.60, 15.00], '-s', label = 'GOLEM_NV')
    axs[0, 1].plot([10, 20, 50], [6.50, 20.90, 59.50], '-h', label = 'GraN-DAG')
    
    #E auPRC
    axs[0, 2].set_title('auPRC')
    axs[0, 2].plot([10, 20, 50], [0.91, 0.92, 0.91], '-o', label = 'Our method')
    axs[0, 2].plot([10, 20, 50], [0.89, 0.92, 0.90], '-^', label = 'NOTEARS')
    axs[0, 2].plot([10, 20, 50], [0.85, 0.88, 0.92], '-s', label = 'GOLEM_NV')
    
    # ER4
    axs[1, 0].set_ylabel('ER4')
    ## auSHDC
    axs[1, 0].plot([10, 20, 50], [3.22, 10.29, 47.12], '-o', label = 'Our method')
    axs[1, 0].plot([10, 20, 50], [2.80, 11.62, 49.65], '-^', label = 'NOTEARS')
    axs[1, 0].plot([10, 20, 50], [5.43, 20.67, 76.64], '-s', label = 'GOLEM_NV')
    
    ## SHD
    axs[1, 1].plot([10, 20, 50], [3.50, 17.60, 94.90], '-o', label = 'Our method')
    axs[1, 1].plot([10, 20, 50], [2.80, 20.70, 117.00], '-^', label = 'NOTEARS')
    axs[1, 1].plot([10, 20, 50], [7.20, 36.80, 149.00], '-s', label = 'GOLEM_NV')
    axs[1, 1].plot([10, 20], [8.40, 28.90], '-h', label = 'GraN-DAG')

    ## auPRC
    axs[1, 2].plot([10, 20, 50], [0.94, 0.89, 0.81], '-o', label = 'Our method')
    axs[1, 2].plot([10, 20, 50], [0.94, 0.87, 0.81], '-^', label = 'NOTEARS')
    axs[1, 2].plot([10, 20, 50], [0.80, 0.62, 0.43], '-s', label = 'GOLEM_NV')

    # SF2
    axs[2, 0].set_ylabel('SF2')
    ## auSHDC
    axs[2, 0].plot([10, 20, 50], [0.55, 2.78, 6.88], '-o', label = 'Our method')
    axs[2, 0].plot([10, 20, 50], [0.54, 2.78, 7.39], '-^', label = 'NOTEARS')
    axs[2, 0].plot([10, 20, 50], [2.10, 6.58, 10.60], '-s', label = 'GOLEM_NV')
    
    ## SHD
    l1 = axs[2, 1].plot([10, 20, 50], [0.40, 4.20, 11.10], '-o', label = 'Our method')[0]
    l2 = axs[2, 1].plot([10, 20, 50], [0.40, 4.20, 11.90], '-^', label = 'NOTEARS')[0]
    l3 = axs[2, 1].plot([10, 20, 50], [2.10, 13.10, 19.00], '-s', label = 'GOLEM_NV')[0]
    l4 = axs[2, 1].plot([10, 20, 50], [11.60, 43.40, 107.80], '-h', label = 'GraN-DAG')[0]

    ## auPRC
    axs[2, 2].plot([10, 20, 50], [0.93, 0.92, 0.95], '-o', label = 'Our method')
    axs[2, 2].plot([10, 20, 50], [0.93, 0.92, 0.94], '-^', label = 'NOTEARS')
    axs[2, 2].plot([10, 20, 50], [0.94, 0.89, 0.42], '-s', label = 'GOLEM_NV')

    # SF4
    axs[3, 0].set_ylabel('SF4')
    ## auSHDC
    axs[3, 0].plot([10, 20, 50], [1.81, 5.39, 20.33], '-o', label = 'Our method')
    axs[3, 0].plot([10, 20, 50], [1.81, 5.36, 21.76], '-^', label = 'NOTEARS')
    axs[3, 0].plot([10, 20, 50], [3.12, 9.21, 99.83], '-s', label = 'GOLEM_NV')
    
    ## SHD
    axs[3, 1].plot([10, 20, 50], [2.40, 7.40, 32.40], '-o', label = 'Our method')
    axs[3, 1].plot([10, 20, 50], [2.40, 7.40, 34.0], '-^', label = 'NOTEARS')
    axs[3, 1].plot([10, 20, 50], [4.20, 13.60, 181.50], '-s', label = 'GOLEM_NV')
    axs[3, 1].plot([10, 20, 50], [11.60, 43.40, 107.80], '-h', label = 'GraN-DAG')

    ## auORC
    axs[3, 2].plot([10, 20, 50], [0.94, 0.94, 0.92], '-o', label = 'Our method')
    axs[3, 2].plot([10, 20, 50], [0.94, 0.94, 0.92], '-^', label = 'NOTEARS')
    axs[3, 2].plot([10, 20, 50], [0.87, 0.84, 0.22], '-s', label = 'GOLEM_NV')

    fig.legend([l1, l2, l3, l4], ["Our method", "NOTEARS", "GOLEM-NV", "GraN-DAG"], loc='lower center', ncol = 4)
    # fig.legend(["Our method", "NOTEARS", "GOLEM-EV", "GraN-DAG"], loc= "lower center", ncol=4)
    fig.tight_layout()
    plt.savefig('homo-NV.png', dpi=500)
    # plt.show()

def plot_hetero():
    fig, axs = plt.subplots(4,3, figsize=(10,9.5), sharex=True)

    # ER2
    axs[0, 0].set_ylabel('ER2')
    ## auSHDC
    axs[0, 0].set_title('auSHDC')
    axs[0, 0].plot([10, 20, 50], [0.87, 2.95, 8.76], '-o', label = 'Our method')
    axs[0, 0].plot([10, 20, 50], [1.54, 2.91, 15.23], '-^', label = 'NOTEARS')
    axs[0, 0].plot([10, 20, 50], [5.89, 12.01, 56.12], '-s', label = 'GOLEM_NV')
    # axs[0, 0].set_xlim([9.5, 50.5])
    axs[0, 0].set_xticks([10, 20, 50])
    
    ## SHD
    axs[0, 1].set_title('SHD(0.3)')
    axs[0, 1].plot([10, 20, 50], [0.70, 4.00, 16.30], '-o', label = 'Our method')
    axs[0, 1].plot([10, 20, 50], [2.30, 8.30, 38.90], '-^', label = 'NOTEARS')
    axs[0, 1].plot([10, 20, 50], [10.50, 21.20, 108.30], '-s', label = 'GOLEM_NV')
    axs[0, 1].plot([10, 20, 50], [17.30, 32.89, 64.75], '-h', label = 'GraN-DAG')
    
    #E auPRC
    axs[0, 2].set_title('auPRC')
    axs[0, 2].plot([10, 20, 50], [0.94, 0.93, 0.89], '-o', label = 'Our method')
    axs[0, 2].plot([10, 20, 50], [0.91, 0.86, 0.83], '-^', label = 'NOTEARS')
    axs[0, 2].plot([10, 20, 50], [0.64, 0.53, 0.32], '-s', label = 'GOLEM_NV')
    
    # ER4
    axs[1, 0].set_ylabel('ER4')
    ## auSHDC
    axs[1, 0].plot([10, 20, 50], [2.40, 13.99, 94.50], '-o', label = 'Our method')
    axs[1, 0].plot([10, 20, 50], [3.90, 20.05, 115.88], '-^', label = 'NOTEARS')
    axs[1, 0].plot([10, 20, 50], [19.50, 47.99, 130.90], '-s', label = 'GOLEM_NV')
    
    ## SHD
    axs[1, 1].plot([10, 20, 50], [2.70, 2.30, 224.60], '-o', label = 'Our method')
    axs[1, 1].plot([10, 20, 50], [5.20, 37.50, 283.40], '-^', label = 'NOTEARS')
    axs[1, 1].plot([10, 20, 50], [33.80, 91.80, 238.00], '-s', label = 'GOLEM_NV')
    axs[1, 1].plot([10], [34.0], '-h', label = 'GraN-DAG')

    ## auPRC
    axs[1, 2].plot([10, 20, 50], [0.95, 0.84, 0.51], '-o', label = 'Our method')
    axs[1, 2].plot([10, 20, 50], [0.90, 0.77, 0.43], '-^', label = 'NOTEARS')
    axs[1, 2].plot([10, 20, 50], [0.34, 0.19, 0.15], '-s', label = 'GOLEM_NV')

    # SF2
    axs[2, 0].set_ylabel('SF2')
    ## auSHDC
    axs[2, 0].plot([10, 20, 50], [0.77, 2.15, 9.73], '-o', label = 'Our method')
    axs[2, 0].plot([10, 20, 50], [1.22, 4.11, 28.20], '-^', label = 'NOTEARS')
    axs[2, 0].plot([10, 20, 50], [2.64, 22.27, 60.71], '-s', label = 'GOLEM_NV')
    
    ## SHD
    l1 = axs[2, 1].plot([10, 20, 50], [0.60, 2.80, 18.20], '-o', label = 'Our method')[0]
    l2 = axs[2, 1].plot([10, 20, 50], [1.60, 7.30, 66.80], '-^', label = 'NOTEARS')[0]
    l3 = axs[2, 1].plot([10, 20, 50], [4.12, 44.50, 94.90], '-s', label = 'GOLEM_NV')[0]
    l4 = axs[2, 1].plot([10, 20], [17.10, 39.71], '-h', label = 'GraN-DAG')[0]

    ## auPRC
    axs[2, 2].plot([10, 20, 50], [0.92, 0.91, 0.88], '-o', label = 'Our method')
    axs[2, 2].plot([10, 20, 50], [0.90, 0.88, 0.66], '-^', label = 'NOTEARS')
    axs[2, 2].plot([10, 20, 50], [0.72, 0.08, 0.04], '-s', label = 'GOLEM_NV')

    # SF4
    axs[3, 0].set_ylabel('SF4')
    ## auSHDC
    axs[3, 0].plot([10, 20, 50], [2.63, 4.30, 29.02], '-o', label = 'Our method')
    axs[3, 0].plot([10, 20, 50], [2.90, 9.26, 51.47], '-^', label = 'NOTEARS')
    axs[3, 0].plot([10, 20, 50], [16.57, 40.20, 97.61], '-s', label = 'GOLEM_NV')
    
    ## SHD
    axs[3, 1].plot([10, 20, 50], [3.40, 5.20, 53.50], '-o', label = 'Our method')
    axs[3, 1].plot([10, 20, 50], [4.00, 14.10, 116.75], '-^', label = 'NOTEARS')
    axs[3, 1].plot([10, 20, 50], [31.20, 76.90, 206.75], '-s', label = 'GOLEM_NV')
    axs[3, 1].plot([10, 20], [28.80, 39.71], '-h', label = 'GraN-DAG')

    ## auORC
    axs[3, 2].plot([10, 20, 50], [0.91, 0.95, 0.78], '-o', label = 'Our method')
    axs[3, 2].plot([10, 20, 50], [0.91, 0.85, 0.68], '-^', label = 'NOTEARS')
    axs[3, 2].plot([10, 20, 50], [0.24, 0.14, 0.06], '-s', label = 'GOLEM_NV')

    fig.legend([l1, l2, l3, l4], ["Our method", "NOTEARS", "GOLEM-EV", "GraN-DAG++"], loc='lower center', ncol = 4)
    # fig.legend(["Our method", "NOTEARS", "GOLEM-EV", "GraN-DAG"], loc= "lower center", ncol=4)
    fig.tight_layout()
    plt.savefig('hetero.png', dpi=500)

if __name__ == '__main__':
    # plot_EV()
    plot_NV()
    # plot_hetero()