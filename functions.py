import xlrd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import time
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.spatial import distance
from statistics import mean
from PIL import Image
from matplotlib import cm
import os
import natsort

class VISHLESHAN():
    def __init__(self):
        pass
    
    def makeDict(self, xlsPath):
        xl_workbook = xlrd.open_workbook(xlsPath)
        sheet_names = xl_workbook.sheet_names()
        metrices = ['ConcBm','BrierBm','p_valueBm','ConcValBm','BrierValBm','PVAlueVal_Bm','ipcwBm']
        results = {key: None for key in sheet_names} 
        for key in results:
            results[key] = {k: None for k in metrices}
        
        for key, value in results.items():
            sheet = pd.read_excel(xlsPath, sheet_name=key)
            for k in value:
                value[k] = sheet[k]

        return results
    
    def plotMetrics(self, dict1, cv, type, metric):
        plt.rc('font', family='Arial')
        fig, axes = plt.subplots(ncols=12, sharey=True, figsize=(10,4))
        fig.subplots_adjust(wspace=0)
        fig.suptitle(str(cv)+'-fold cross-validated testing results',fontweight = 'bold',  fontsize=16, y = 0.95)
        axes[0].set_ylabel('Value', fontsize = 14)
        labels = ['mrna',
            'meth',
            'mirna',
            'mrna\n+\nmeth',
            'mrna\n+\nmirna',
            'mrna\n+\nmeth\n+\nmirna',
            'mrna\n+\nclin',
            'meth\n+\nclin',
            'mirna\n+\nclin',
            'mrna\n+\nmeth\n+\nclin',
            'mrna\n+\nmirna\n+\nclin',
            'mrna\n+\nmeth\n+\nmirna\n+\nclin']

        for ax, name1, label in zip(axes, list(dict1), labels):
    
            #box1 = ax.boxplot([dict1[name1][item].sort_values(ascending = False).reset_index(drop=True)[0:cv] if item=='ConcValBm' else dict1[name1][item].sort_values(ascending = True).reset_index(drop=True)[0:cv] for item in ['ConcValBm', 'BrierValBm', 'ipcwBm']], positions= [0.9, 1.9], widths = 0.6, patch_artist = True)
            if metric=='BrierValBm':
                box1 = ax.boxplot([dict1[name1][item].sort_values(ascending = True).reset_index(drop=True)[0:cv] for item in [metric]], widths = 0.5, patch_artist = True)
            else:
                box1 = ax.boxplot([dict1[name1][item].sort_values(ascending = False).reset_index(drop=True)[0:cv] for item in [metric]], widths = 0.5, patch_artist = True)

            if metric=='BrierValBm':
                colors1 = ['limegreen']
            elif metric=='ConcValBm':
                colors1 = ['darkblue']
            elif metric=='ipcwBm':
                colors1=['darkmagenta']

            for patch1, color1 in zip(box1['boxes'], colors1):
                patch1.set_facecolor(color1)

            if metric=='BrierValBm':
                ax.text(0.58, 0.38, u"\u03bc:"+str(np.round(np.mean(dict1[name1][metric].sort_values(ascending = True).reset_index(drop=True)[0:cv]),2)), fontsize=13)
            else:  
                ax.text(0.58, 0.41, u"\u03bc:"+str(np.round(np.mean(dict1[name1][metric].sort_values(ascending = False).reset_index(drop=True)[0:cv]),2)), fontsize=13)

            # if metric=='BrierValBm':
            #     ax.set_xticklabels([u"\u03bc:"+str(np.round(np.mean(dict1[name1][metric].sort_values(ascending = True).reset_index(drop=True)[0:cv]),2))], rotation=45)
            # else:
            #     ax.set_xticklabels([u"\u03bc:"+str(np.round(np.mean(dict1[name1][metric].sort_values(ascending = False).reset_index(drop=True)[0:cv]),2))], rotation=45)
            ax.margins(0.05)
            ax.tick_params(labelbottom=False)
            # ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14) 
            ax.grid(linestyle='-.', linewidth=0.9)
            if metric=='BrierValBm':
                ax.set_ylim([0,0.5])
                ax.set_yticks(np.arange(0, 0.5, 0.1))
            else:
                ax.set_ylim([0.4,1])
                ax.set_yticks(np.arange(0.4, 1, 0.1))
            ax.set_xlabel(label, fontsize = 14)
        ax.legend([box1["boxes"][0]], [metric+': TSNE('+type+')'], bbox_to_anchor=(1, 1.02, 0, -0.01),ncol=2, prop={'size': 14})

        #print(len(box1))
        transFigure = fig.transFigure.inverted()
        coord=[0,0]
        for i in range(len(list(dict1))):
            if metric=='BrierValBm':
                coord=np.vstack([coord,transFigure.transform(axes[i].transData.transform([1,np.median(dict1[list(dict1)[i]][metric].sort_values(ascending = True).reset_index(drop=True)[0:cv])]))])
            else:
                coord=np.vstack([coord,transFigure.transform(axes[i].transData.transform([1,np.median(dict1[list(dict1)[i]][metric].sort_values(ascending = False).reset_index(drop=True)[0:cv])]))])

        coord=np.delete(coord, 0, 0)
        line1 = matplotlib.lines.Line2D((coord[0][0],coord[1][0]),(coord[0][1],coord[1][1]),transform=fig.transFigure)
        line2 = matplotlib.lines.Line2D((coord[1][0],coord[2][0]),(coord[1][1],coord[2][1]),transform=fig.transFigure)
        line3 = matplotlib.lines.Line2D((coord[2][0],coord[3][0]),(coord[2][1],coord[3][1]),transform=fig.transFigure)
        line4 = matplotlib.lines.Line2D((coord[3][0],coord[4][0]),(coord[3][1],coord[4][1]),transform=fig.transFigure)
        line5 = matplotlib.lines.Line2D((coord[4][0],coord[5][0]),(coord[4][1],coord[5][1]),transform=fig.transFigure)
        line6 = matplotlib.lines.Line2D((coord[5][0],coord[6][0]),(coord[5][1],coord[6][1]),transform=fig.transFigure)
        line7 = matplotlib.lines.Line2D((coord[6][0],coord[7][0]),(coord[6][1],coord[7][1]),transform=fig.transFigure)
        line8 = matplotlib.lines.Line2D((coord[7][0],coord[8][0]),(coord[7][1],coord[8][1]),transform=fig.transFigure)
        line9 = matplotlib.lines.Line2D((coord[8][0],coord[9][0]),(coord[8][1],coord[9][1]),transform=fig.transFigure)
        line10 = matplotlib.lines.Line2D((coord[9][0],coord[10][0]),(coord[9][1],coord[10][1]),transform=fig.transFigure)
        line11= matplotlib.lines.Line2D((coord[10][0],coord[11][0]),(coord[10][1],coord[11][1]),transform=fig.transFigure)

        fig.lines = line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,
        fig.savefig('plots/perf_'+metric+'_'+str(cv)+'CV_'+type+'.png', format = 'png', dpi = 1200, bbox_inches='tight')
        plt.show()


    def plotCompare(self, dict1, dict2, cv, type):
        plt.rc('font', family='Arial')
        fig, axes = plt.subplots(ncols=12, sharey=True, figsize=(10,5))
        fig.subplots_adjust(wspace=0)
        #fig.suptitle('5-fold cross-validated '+type+' model results 10CV',fontweight = 'bold',  fontsize=16, y = 0.92)
        fig.suptitle(str(cv)+'-fold cross-validated testing results',fontweight = 'bold',  fontsize=16, y = 0.93)

        axes[0].set_ylabel('Value', fontsize = 14)
        labels = ['mrna',
            'meth',
            'mirna',
            'mrna\n+\nmeth',
            'mrna\n+\nmirna',
            'mrna\n+\nmeth\n+\nmirna',
            'mrna\n+\nclin',
            'mirna\n+\nclin',
            'meth\n+\nclin',
            'mrna\n+\nmeth\n+\nclin',
            'mrna\n+\nmirna\n+\nclin',
            'mrna\n+\nmeth\n+\nmirna\n+\nclin']

        for ax, name1,name2, label in zip(axes, list(dict1), list(dict2), labels):

            box1 = ax.boxplot([dict1[name1][item].sort_values(ascending = False).reset_index(drop=True)[0:cv] if item=='ConcValBm' else dict1[name1][item].sort_values(ascending = True).reset_index(drop=True)[0:cv] for item in ['ConcValBm', 'BrierValBm']], positions= [0.9, 1.9], widths = 0.6, patch_artist = True)
            box2 = ax.boxplot([dict2[name2][item].sort_values(ascending = False).reset_index(drop=True)[0:cv] if item=='ConcValBm' else dict2[name2][item].sort_values(ascending = True).reset_index(drop=True)[0:cv] for item in ['ConcValBm', 'BrierValBm']], positions= [1.1, 2.1], widths = 0.6, patch_artist = True)

            colors1 = ['limegreen', 'limegreen']
            colors2 = ['blueviolet', 'blueviolet']
            for patch1, patch2, color1, color2 in zip(box1['boxes'], box2['boxes'], colors1, colors2):
                patch1.set_facecolor(color1)
                patch2.set_facecolor(color2)

            #ax.text(0.5, 0.9, label)
            ax.set_xticklabels(['C-Index', 'Brier'], rotation=45)
            ax.margins(0.05)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=14) 
            ax.set_yticks(np.arange(0, 1, 0.1))
            ax.grid(linestyle='-.', linewidth=0.9)
            ax.set_ylim([0,1])
            ax.set_xlabel(label, fontsize = 14, fontweight='bold')
        ax.legend([box1["boxes"][0], box2["boxes"][0]], ['TSNE('+type+')', 'UMAP('+type+')'], bbox_to_anchor=(1, 1.02, 0, -0.01),ncol=2, prop={'size': 12})
        #fig.savefig('plots/perf_'+str(cv)+'CV_'+type+'.png', format = 'png', dpi = 1200, bbox_inches='tight')
        plt.show()

    def dichot(self, T, F, surv_prob, median):
        T1 = T[surv_prob >= median]
        T2 = T[surv_prob < median]
        E1 = F[surv_prob >= median]
        E2 = F[surv_prob < median]
        result = logrank_test(T1, T2, E1, E2)
        p = result.p_value
        return T1, T2, E1, E2, p


    def plotKM(self, T, surv_prob, F, year, train_val, median, breaks):
        T1, T2, E1, E2, p = self.dichot(T, F, surv_prob, median)

        plt.rc('font', family='Arial')
        fig, ax = plt.subplots(ncols=1, figsize=(6,6))
        #plt.figure(figsize=(12,4))
        #plt.subplot(1,2,1)
        days_plot = 9*365

        kmf = KaplanMeierFitter()
        for i in range(2):
            if i==0:
                kmf.fit(T1.tolist(),E1.tolist())
                kmf.plot(color='darkgreen')
            if i==1:
                kmf.fit(T2.tolist(),E2.tolist())
                kmf.plot(color='darkred')
        N1='N='+ str(len(T1))
        N2='N='+ str(len(T2))

        ax.set_xticks(np.arange(0, days_plot, 365))
        ax.set_yticks(np.arange(0, 1.125, 0.125))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim([0, days_plot])
        ax.set_ylim([0,1])
        ax.text(50, 0.025, 'logrank p-value = ' +str('%.3g'%(p)), bbox=dict(facecolor='red', alpha=0.3), fontsize=14)

        ax.set_xlabel('Follow-up time (days)', fontsize = 15)
        ax.set_ylabel('Probability of survival', fontsize = 15)
        ax.legend(['Low Risk Individuals ' + N1 ,'High Risk Individuals ' + N2 ], fontsize=12, loc='upper right')
        ax.set_title('%s set Kaplan-Meier Curves'%(train_val), fontweight = 'bold', fontsize = 15)
        ax.grid(linestyle='-.', linewidth=0.9)
        
        for spine in ax.spines:
            ax.spines[spine].set_linewidth(2)

        fig.savefig('plots/KM_'+train_val+'.png', format = 'png', dpi = 1200, bbox_inches='tight')
        plt.show()
        return None

    def minimum_bounding_rectangle(self, points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an nx2 matrix of coordinates
        """
        from scipy.ndimage.interpolation import rotate
        pi2 = np.pi/2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-pi2),
            np.cos(angles+pi2),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval
    
    def deepinsight(self, path):
        df_subset = pd.read_csv(path) #ALGO output

        mbr = self.minimum_bounding_rectangle(df_subset.values)

        # Calculate tilit of bounding box
        y2,y1,x2,x1 = mbr[0][1],mbr[1][1],mbr[0][0],mbr[1][0]
        theta = (y2-y1)/(x2-x1)
        angle  = math.degrees(np.arctan(theta))
        angle  = np.arctan(theta)
        r_matrix = np.asarray([[math.cos(angle),-1*math.sin(angle)],[math.sin(angle),math.cos(angle)]])

        zrect = np.matmul(mbr,r_matrix)  
        z = np.asarray(np.matmul(df_subset,r_matrix))
        rz_subset  = pd.DataFrame()
        rz_subset['rot-tsne-2d-one'] = z[:,0]
        rz_subset['rot-tsne-2d-two'] = z[:,1]
        #rz_subset.to_csv('tsne_515_seed22/deepinsight_outputs/mrna515_5_cosine_rotated.csv')

        z_dist = distance.cdist(z, z, 'euclidean')
        min_z_dist = z_dist[z_dist>0].min()

        rec_x_axis,rec_y_axis = abs(zrect[0][0] - zrect[1][0]),abs(zrect[1][1] - zrect[2][1])

        precision_old = math.sqrt(2)
        A = math.ceil((rec_x_axis*precision_old)/min_z_dist)
        B = math.ceil((rec_y_axis*precision_old)/min_z_dist)

        max_pix_size = 120
        if (max([A,B]) > max_pix_size):
            precision = (precision_old*max_pix_size/max([A,B]))
            A = math.ceil((rec_x_axis*precision)/min_z_dist)
            B = math.ceil((rec_y_axis*precision)/min_z_dist)

        x_coord,y_coord = rz_subset.iloc[:,0].values,rz_subset.iloc[:,1].values
        x_min,y_min,x_max,y_max = min(x_coord),min(y_coord),max(x_coord),max(x_coord)
        x_pixel = (1 + (A*(x_coord - x_min))/(x_max - x_min))
        y_pixel = (1 + (B*(y_coord - y_min))/(y_max - y_min))
        round_x_pixel = np.array([int(np.round(x)) for x in x_pixel])
        round_y_pixel = np.array([int(np.round(y)) for y in y_pixel])

        pix_subset  = pd.DataFrame()
        pix_subset['pix-tsne-2d-one'] = round_x_pixel
        pix_subset['pix-tsne-2d-two'] = round_y_pixel
        #pix_subset.to_csv('tsne_515_seed22/deepinsight_outputs/mrna515_6_cosine_pix_coord.csv')

        unique_coord = self.getOverlappingGeneID(pix_subset)

        return pix_subset, unique_coord

    def getOverlappingGeneID(self, pix_df):
        pixels = []
        for index in pix_df.index:
            px = tuple(pix_df.loc[index].tolist())
            pixels.append(px)
        
        unique_coord = dict.fromkeys(list(set(pixels)))
        for index in pix_df.index:
            px = tuple(pix_df.loc[index].tolist())
            if unique_coord[px] == None:
                unique_coord[px] = [index]
            else:
                unique_coord[px].append(index)

        return unique_coord

        # pixel_set = []
        # for _ in range(len(round_x_pixel)):
        #     pixel_set.append((round_x_pixel[_],round_y_pixel[_]))

        # unique_coord = {}
        # for i in range(len(pixel_set)):
        #     unique_coord[pixel_set[i]] = self.list_duplicates_of(pixel_set,pixel_set[i])
        
    def average_pix(self, patient_list, datapx):
        c = np.zeros(np.load(datapx+patient_list[0]+'.npy').shape)
        for patient in patient_list:
            image = np.load(datapx+patient+'.npy')
            c = c+image
        c = c/len(patient_list)
        return c       

    #function to find overlapping points
    def list_duplicates_of(self, seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item,start_at+1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs