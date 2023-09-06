import numpy as np
import pandas as pd
import math
from skimage.transform import hough_line, hough_line_peaks

from aicssegmentation.core.vessel import filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, disk, dilation
from skimage import draw
from skimage.draw import polygon, polygon2mask
from skimage.transform import probabilistic_hough_line
from skimage.measure import profile_line
from skimage.morphology import erosion, binary_erosion, binary_dilation, opening, closing, disk, skeletonize
from skimage.segmentation import expand_labels
from skimage.filters import meijering
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

################################################################################################################################

# create a mask out of vertices
def create_mask_from_shapes(vertices_polygons, im_shape):

    '''
    Function to create mask from vertices of the polygons. It removes regions of overlap.

    Input:
    - vertices_polygons - list of polygons (coordinates of vertices)
    - im_shape - size of the image to create
    Output:
    - label image of polygons
    '''

    # create a mask out of vertices
    mask = np.zeros(im_shape).astype('uint8')

    for i,poly in enumerate(vertices_polygons):

        # if drawing was in 3D
        if len(poly.shape) > 2:
            mask_coord = polygon(vertices_polygons[i][:,1],vertices_polygons[i][:,2],shape=im_shape)
        else:
            #mask_coord = polygon(vertices_polygons[i][:,0],vertices_polygons[i][:,1],shape=im_shape)
            mask_single = polygon2mask(im_shape,vertices_polygons[i])

        # it's iteratively adding, so regions with anly overlap will be higher than i+1
        mask[mask_single] = mask[mask_single] + (i+1)

        # mark areas of the overlap
        mask[mask > (i+1)] = 255

    mask_shapes_overlap = mask

    # shapes without overlapping regions
    mask_shapes = mask_shapes_overlap.copy()
    mask_shapes[mask_shapes == 255] = 0

    return mask_shapes_overlap,mask_shapes

################################################################################################################################

# segmentation of actin in 3D volumes
def segment_actin_3D(image_actin):

    '''
    Function wrapping Allen Cell Segmenter algorithm for fibers.
    Note that it will segment differently single cells and entire field of view because of the normalization steps.
    Based on: https://github.com/AllenCell/aics-segmentation/blob/main/lookup_table_demo/playground_filament3d.ipynb 

    Input:
    image_actin - 3D image of actin 

    Output:
    image_actin_mask - 3D segmented fibers
    '''

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    f3_param = [[1, 0.01]]
    ################################

    # intensity normalization
    struct_img = intensity_normalization(image_actin, scaling_param=intensity_scaling_param)

    # smoothing with edge preserving smoothing 
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    # segmentation
    image_actin_mask = filament_3d_wrapper(structure_img_smooth, f3_param)

    return image_actin_mask

################################################################################################################################

def find_fibers_orientation(image_actin_mask_2D):

    '''
    Function that uses Hough transform to find a dominant orientation of fibers in a 2D image of single cells.

    Input:
    image_actin_mask_2D

    Output:
    dominant_flow - angle (radians) 
    '''

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image_actin_mask_2D, theta=tested_angles)

    _, angle_array, dist_array = hough_line_peaks(h, theta, d)

    dominant_flow = np.mean(angle_array[:4])

    return dominant_flow

################################################################################################################################
'''
def calculate_orientation(p0,p1):

    dc = p1[0] - p0[0]
    dr = -(p1[1] - p0[1])
    
    myrad = np.arctan2(dr, dc) 
    if myrad < 0:
        myrad = np.pi + myrad
    
    return myrad
'''

def calculate_orientation(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.arctan2(dy, dx)


################################################################################################################################

def find_fibers_orientation_v2(actin_im):

    skeleton_im = meijering(actin_im,sigmas=range(1, 3),black_ridges=False)
    skeleton_im[skeleton_im < 0.5] = 0

    # find straight lines in the image
    lines = probabilistic_hough_line(skeleton_im, threshold=1, line_length=15,
                                    line_gap=0)

    # calculate orientation of the lines
    rad_list = []
    for line in lines:

        p0, p1 = line
        #p0 = [p0[1],p0[0]]
        #p1 = [p1[1],p1[0]]
        actin_rad = calculate_orientation(p1,p0)

        rad_list.append(actin_rad)

    return lines,rad_list

################################################################################################################################

def orientations_from_vertices(vert):

    '''
    It accepts vertices in the format given by Napari
    '''

    my_rad_list = []

    for i in range(len(vert)):

        p0 = vert[i,:]
        
        if i == (len(vert)-1):
            p1 = vert[0,:]
        else:
            p1= vert[i+1,:]

        # change of the coordinates to match orientation calculated for actin fibers
        p0 = [p0[1],p0[0]]
        p1 = [p1[1],p1[0]]
        my_rad = calculate_orientation(p1,p0)

        # old version
        # my_rad = -(calculate_orientation(p1,p0) % np.pi - np.pi/2)

        my_rad_list.append(my_rad)

    return my_rad_list

################################################################################################################################

def signal_from_vertices(vert,signal_im,line_width=1,**kwargs):

    '''
    Calculates signal at the perimeter using vertices.

    '''

    signal_line = []

    for i in range(len(vert)):

        p0 = vert[i,:]
        
        if i == (len(vert)-1):
            p1 = vert[0,:]
        else:
            p1= vert[i+1,:]

        signal_segment = profile_line(signal_im,p0,p1,line_width,**kwargs)

        signal_line.extend(signal_segment)

    return signal_line

#################################################################################################################################

def divide_cell_outside_ring(cell_image,cell_center,ring_thickness,segment_number):

    # generate single pixel line inside the desired ring
    eroded_image_1 = erosion(cell_image,disk(int(ring_thickness)))
    eroded_image_2 = erosion(eroded_image_1,disk(1))

    seed_perim_image = eroded_image_1.astype(int) - eroded_image_2.astype(int)

    # calculate seeds for clustering
    t = np.nonzero(seed_perim_image)
    points_array = np.array(t).T

    clustering = KMeans(n_clusters=segment_number).fit(points_array)

    center_point_list = []

    for i in range(segment_number):

        center_point = np.mean(points_array[clustering.labels_==i,:],axis=0)
        center_point_list.append(center_point)

    center_point_array = np.array(center_point_list)

    # calculate where points from the ring belong
    eroded_image = erosion(cell_image,disk(ring_thickness))
    image_ring = cell_image - eroded_image.astype(int)
    t = np.nonzero(image_ring)
    points_array = np.array(t).T

    dist_mat = distance_matrix(points_array,center_point_array)

    cluster_identity = np.argmin(dist_mat,axis=1)
    cluster_identity = np.expand_dims(cluster_identity,axis=1)

    # concatenate points position with their cluster identity
    points_array = np.concatenate((points_array,cluster_identity),axis=1)

    ##########################################################################
    # arrange clockwise from most right

    # define centroid array
    centroid_array = pd.DataFrame([np.mean(points_array[points_array[:,2]==x,:],axis=0) for x in range(segment_number)],columns=['x','y','set'])
    centroid_array['centroid_angle'] = [np.arctan2(x-cell_center[1],y-cell_center[0]) for x,y in zip(centroid_array.loc[:,'x'],centroid_array.loc[:,'y'])]

    # shift to start from the most left region
    centroid_array.loc[centroid_array.centroid_angle<0,'centroid_angle'] = centroid_array.loc[centroid_array.centroid_angle<0,'centroid_angle'] + 2*np.pi

    # sort
    centroid_array = centroid_array.sort_values('centroid_angle',ignore_index=True)

    # change identity based on the new order
    points_array[:,2] = [centroid_array.loc[centroid_array.set == x,:].index[0] for x in points_array[:,2]]

    return points_array
     
#################################################################################################################################

def fill_gaps_between_cells(mask_shapes_overlap):

    '''
    input:
        - label image of polygons without overlapping regions
    output:
        - contested regions divided between the cells  
    '''

    # find narrow passages between the cells
    # it's defined as points that are within 8 px from a cell if they are simultaneously within 10px from another cell + morphological rearrangements to make it smoother

    mask_shapes = mask_shapes_overlap.copy()
    mask_shapes[mask_shapes == 255] = 0

    mask_list_small = []
    mask_list_big = []

    # loop through the objects 
    for i in range(np.max(mask_shapes)):

        mask = (mask_shapes == i+1)

        mask_dilated_small = binary_dilation(mask,disk(8))
        mask_dilated_big = binary_dilation(mask,disk(10))

        mask_list_small.append(mask_dilated_small.astype(int) - mask)
        mask_list_big.append(mask_dilated_big.astype(int) - mask)


    # combine the masks (choose regions that may contain corrections)
    possible = np.sum(np.array(mask_list_big),axis=0)>1

    # select which regions from the small rings are in the possible territory
    # keeps regions in small distance to a cell not further than the bigger distance from another cell
    t = np.logical_and(np.array(mask_list_small),possible)
    passages = np.sum(np.array(t),axis=0)

    # trim the passages
    mask_to_trim = ((mask_shapes_overlap > 0) | (passages > 1))
    mask_trimmed = ((opening(mask_to_trim,disk(10))) | (mask_shapes_overlap)>0)
    mask_trimmed = ((binary_erosion(mask_trimmed,disk(5))) | (mask_shapes_overlap)>0)
    mask_trimmed = ((closing(mask_trimmed,disk(2))) | (mask_shapes_overlap)>0)

    # combine the pixels that need to be re-assigned
    to_divide = mask_trimmed.astype(int) - (mask_shapes_overlap > 0).astype(int) + (mask_shapes_overlap==255).astype(int)

    # re-assign pixels
    im_divided = expand_labels(mask_shapes,250)*(to_divide).astype(int)

    return im_divided

#################################################################################################################################
'''
def calculate_perpendicular_index(angle_actin,angle_membrane):

    # a value between 0 and 1
    # 0 - parallel
    # 1 - perpendicular
    
    orientation = np.abs(np.abs(np.abs(angle_membrane - angle_actin) - (np.pi/2))/(np.pi/2)-1)

    return orientation
'''

def calculate_perpendicular_index(alpha, beta):

    # Calculate the angle between lines
    angle_between_lines = abs(alpha - beta)
    
    # Get values between 0 and pi
    if angle_between_lines > math.pi:
        angle_between_lines = angle_between_lines - math.pi

    # Get values between 0 and pi/2
    if angle_between_lines > (math.pi/2):
        angle_between_lines = math.pi/2 - (angle_between_lines - math.pi/2)

    M = angle_between_lines/(math.pi/2)

    return M

'''
# Test the function with some examples
print(angle_measure(math.pi/4, -math.pi/4))  # Should be close to 1 (perpendicular)
print(angle_measure(math.pi/4, math.pi/4))  # Should be close to 0 (parallel)
print(angle_measure(0, math.pi))           # Should be close to 0 (parallel)
print(angle_measure(math.pi/3, math.pi/6)) # Should be between 0 and 1
'''

#################################################################################################################################
def sk_line_profile_coordinates(src, dst, linewidth=1):

    """
    
    https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/measure/profile.py#L7-L120

    Return the coordinates of the profile of an image along a scan line.


    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])


#################################################################################################################################
def get_internal_points(vert,line_width=3):

    '''
    Calculates coordinates of points internal to the polygon based on vertices.
    This function doesn't exclude points that happen to be outside of the polygon (happens at the corners).
    It also doesn't segregate points that are claimed by more than one point at the perimeter.

    '''

    if line_width < 3:
        return 'Error - line width has to be at least 3.'

    coord_line_x = []
    coord_line_y = []

    for i in range(len(vert)):

        p0 = vert[i,:]
        
        if i == (len(vert)-1):
            p1 = vert[0,:]
        else:
            p1= vert[i+1,:]


        signal_segment = sk_line_profile_coordinates(p0,p1,linewidth=line_width)

        coord_line_x.extend(signal_segment[0])
        coord_line_y.extend(signal_segment[1])

    # reshape output
    t = np.array([coord_line_x,coord_line_y]).T.astype(int)

    # layers are organized from outside to inside 
    start_including = int(line_width/2)

    df_list = []
    for i in range(start_including,line_width):

        layer = t[i,...]
        l_unique = np.unique(layer.astype(int),axis=0,return_index=True)

        df = pd.DataFrame(layer)
        df['unique'] = False
        df['layer'] = i - start_including
        df['ord'] = df.index
        df.loc[l_unique[1],'unique'] = True 

        df_list.append(df)

    df_all = pd.concat(df_list,ignore_index = True)
    df_all.columns = ['r','c','uniqe','layer','ord']

    return df_all

#################################################################################################################################

def create_edge_visual(im_size,vertices,orientations,line_width=5):

    '''
    Utility to create an image visualizing edge orientation vs main flow of actin in single cells.
    '''

    # prepare image of edge orientation vs actin
    edge_image = np.zeros(im_size)

    for i,o in zip(range(len(vertices)),orientations):
        
        p0 = vertices[i]

        if (i == (len(vertices)-1)):
            p1 = vertices[0]
        else:
            p1 = vertices[i+1]

        rr,cc = draw.line(p0[0],p0[1],p1[0],p1[1])
        edge_image[rr,cc] = o

    edge_image = dilation(edge_image,disk(line_width))

    return edge_image

#################################################################################################################################

def interpolate_and_fill(signal):

    # where signal is missing
    x = np.argwhere((np.isnan(signal))).squeeze()

    # existing signal
    xp = np.argwhere(~(np.isnan(signal))).squeeze()
    fp = signal[~(np.isnan(signal))].squeeze()
    
    # interpolate the values
    values_interp = np.interp(x,xp,fp)
    
    # fill in the values
    signal[x] = values_interp

    return signal

#################################################################################################################################