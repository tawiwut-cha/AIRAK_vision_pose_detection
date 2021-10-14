'''
Use OpenPose with video

Tawiwut C.
'''

# imports
import cv2
import os
from urllib.request import urlretrieve

# Constants
DATASET_PATH = os.path.join('datasets')
MODEL_PATH = os.path.join('model')
PROTO_FILE_NM = "mpi.prototxt"
WEIGHTS_FILE_NM = "mpi.caffemodel"
NPOINTS = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
NET = cv2.dnn.readNetFromCaffe(os.path.join(MODEL_PATH, PROTO_FILE_NM), os.path.join(MODEL_PATH, WEIGHTS_FILE_NM))


# # helper functions
def load_model(model_path=MODEL_PATH, proto_file_nm=PROTO_FILE_NM, weights_file_nm=WEIGHTS_FILE_NM):
    print('Checking installed models....')
    if not os.path.isdir(model_path):
        print('Creating model directory...')
        os.mkdir("model")
    
    proto_file_path = os.path.join(model_path, proto_file_nm)
    if not os.path.isfile(proto_file_path):
        # Download the proto file
        print('Downloading proto file...')
        urlretrieve(
            'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt',
            proto_file_path
            )

    weights_file_path = os.path.join(model_path, weights_file_nm)
    if not os.path.isfile(weights_file_path):
        # Download the model file
        print('Downloading weights file (this may take a while)...')
        urlretrieve(
            'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel',
            weights_file_path
            )
    print('Done')

def stream_video(source_or_filename=0, dataset_path=DATASET_PATH, frame_processing=None, **kwargs):
    '''
    Streams video from either webcam or video file
    Press escape key to stop streaming
    
    Params: 
    source_or_filename - ex: 0 for webcam or 'filname' for video file
    frame_processing - function to process a frame
    dataset_path - filepath
    **kwargs - keyword arguments for frame processing
    '''
    assert isinstance(source_or_filename, (int, str)), 'source_or_filename must be int or str'
    
    if isinstance(source_or_filename, int):  # index to a system's available camera is passed
        s = source_or_filename
    if isinstance(source_or_filename, str):  # filepath is passed
        s = os.path.join(dataset_path, source_or_filename)

    source = cv2.VideoCapture(s)
    fps = int(source.get(cv2.CAP_PROP_FPS))
    
    win_name = 'Video stream'
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    while cv2.waitKey(fps) != 27:  # Escape key for exit, if frame_processing takes long the video playback will be delayed
        has_frame, frame = source.read()
        if not has_frame:
            break
        if frame_processing:
            frame = frame_processing(frame, **kwargs)
        cv2.imshow(win_name, frame)
    
    source.release()
    cv2.destroyWindow(win_name)

def apply_open_pose(im, net=NET, nPoints=NPOINTS, pose_pairs=POSE_PAIRS):
    '''
    Apply open pose model to an image or frame
    frame - image or frame
    net - loaded deep learning model
    nPoints - number of output joints to try to draw
    '''
    inHeight, inWidth = im.shape[0], im.shape[1]
    # convert image to blob to feed into model
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    # Forward Pass
    output = net.forward()
    
    # Extract points
    ## X and Y Scale
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]

    ## Empty list to store the detected keypoints
    points = []

    ## Confidence treshold 
    threshold = 0.1

    for i in range(nPoints):
        ## Obtain probability map
        probMap = output[0, i, :, :]
        
        ## Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        ## Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold : 
            ## Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in pose_pairs:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(im, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(im, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    return im
    


if __name__ == '__main__':
    load_model()
    stream_video(0, frame_processing=apply_open_pose)
    # stream_video('pose_0.mp4', frame_processing=apply_open_pose)
    
    # try out apply_open_pose to image
    # still_img = cv2.imread('datasets/Tiger_Woods.png')
    # cv2.imshow('still img', still_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # still_img_open_pose = apply_open_pose(still_img)
    # cv2.imshow('still img', still_img_open_pose)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  




