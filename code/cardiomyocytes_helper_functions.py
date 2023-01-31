import numpy as np
import math
from skimage.transform import hough_line, hough_line_peaks
import sys
sys.path.append('/Users/sahanabalasubramanian/Documents/Sahana_Kasia_CBI_image_analysis/aics-segmentation')
from aicssegmentation.core.vessel import filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects  
from skimage.draw import polygon
from skimage.transform import probabilistic_hough_line
from skimage.measure import profile_line
from skimage.morphology import erosion, binary_erosion, binary_dilation, opening, closing, disk, skeletonize
from skimage.segmentation import expand_labels
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

################################################################################################################################
# create a mask out of vertices
def create_mask_from_shapes(vertices_polygons, im_shape):

    '''
    Input:
    - vertices_polygons - list of polygons (coordinates of vertices)
    - im_shape - ize of the image to create
    Output:
    - label image of polygons
    '''

    mask = np.zeros(im_shape).astype('uint8')

    for i,poly in enumerate(vertices_polygons):

        # if drawing was in 3D
        if len(poly.shape) > 2:
            mask_coord = polygon(vertices_polygons[i][:,1],vertices_polygons[i][:,2],shape=im_shape)
        else:
            mask_coord = polygon(vertices_polygons[i][:,0],vertices_polygons[i][:,1],shape=im_shape)

        mask[mask_coord] = i+1

    return mask

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
def calculate_orientation(p0,p1):
    
    myrad = -(math.atan2(p1[1]-p0[1], p1[0]-p0[0]) + np.pi/2)
    
    return myrad

################################################################################################################################
def find_fibers_orientation_v2(actin_im):

    # skeletonize the image
    skeleton_im = skeletonize(actin_im)

    # find straight lines in the image
    lines = probabilistic_hough_line(skeleton_im, threshold=4, line_length=15,
                                    line_gap=2)

    # calculate orientation of the lines
    rad_list = []
    for line in lines:

        p0, p1 = line
        actin_rad = calculate_orientation(p0,p1)

        rad_list.append(actin_rad)

    return lines,rad_list


################################################################################################################################

def orientations_from_vertices(vert):

    '''
    It accepts vertices in the format given by Napari
    Returns the list of orientations of membrane segments in the style of scikit.image orientation (-pi/2 to pi/2)
    '''

    my_rad_list = []

    for i in range(len(vert)):

        p0 = vert[i,:]
        
        if i == (len(vert)-1):
            p1 = vert[0,:]
        else:
            p1= vert[i+1,:]

        my_rad = -(calculate_orientation(p1,p0) % np.pi - np.pi/2)

        my_rad_list.append(my_rad)

    return my_rad_list

################################################################################################################################

def signal_from_vertices(vert,signal_im):

    '''
    It accepts vertices in the format given by Napari
    '''

    signal_line = []

    for i in range(len(vert)):

        p0 = vert[i,:]
        
        if i == (len(vert)-1):
            p1 = vert[0,:]
        else:
            p1= vert[i+1,:]

        signal_segment = profile_line(signal_im,p0,p1)

        signal_line.extend(signal_segment)

    return signal_line

#################################################################################################################################

def divide_cell_outside_ring(cell_image,ring_thickness,segment_number):

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

    return points_array
     
#################################################################################################################################

def fill_gaps_between_cells(mask_shapes_overlap):

    # find narrow passages between the cells
    # it's defined as points that are within 8 px from a cell if they are simultaneously within 10px from another cell + morphological rearrangements to make it smoother

    mask_shapes = mask_shapes_overlap.copy()
    mask_shapes[mask_shapes == 255] = 0

    mask_list_small = []
    mask_list_big = []

    for i in range(np.max(mask_shapes)):

        mask = (mask_shapes == i+1)

        mask_dilated_small = binary_dilation(mask,disk(8))
        mask_dilated_big = binary_dilation(mask,disk(10))

        mask_list_small.append(mask_dilated_small.astype(int) - mask)
        mask_list_big.append(mask_dilated_big.astype(int) - mask)


    # combine the masks
    possible = np.sum(np.array(mask_list_big),axis=0)>1

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