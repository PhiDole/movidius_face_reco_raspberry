import cv2
import numpy as np
import argparse
from armv7l.openvino.inference_engine import IENetwork, IEPlugin
"""
parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                    'The sample and model are simplified and could be used for a single person on the frame.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--proto', help='Path to .prototxt')
parser.add_argument('--model', help='Path to .caffemodel')
parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                      'It could be (COCO, MPI, HAND) depends on dataset.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
parser.add_argument('--scale', default=0.003922, type=float, help='Scale for blob.')

args = parser.parse_args()

if args.dataset == 'COCO':"""
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
"""
elif args.dataset == 'MPI':
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
else:
    assert(args.dataset == 'HAND')
    BODY_PARTS = { "Wrist": 0,
                   "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                   "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                   "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
                   "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                   "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
                 }

    POSE_PAIRS = [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                   ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                   ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                   ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                   ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                   ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                   ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                   ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                   ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                   ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]

"""
inWidth = 456
inHeight = 256
inScale = 1.0
thr = 0.1

file_xml = "/home/pi/Documents/model_project/intel_models/human_pose_estimation/FP16/human-pose-estimation-0001.xml"
file_bin = "/home/pi/Documents/model_project/intel_models/human_pose_estimation/FP16/human-pose-estimation-0001.bin"
plugin = IEPlugin(device="MYRIAD", plugin_dirs=None)
net = IENetwork(model=file_xml, weights=file_bin)
#net = cv2.dnn.readNet(cv2.samples.findFile(file_xml), cv2.samples.findFile(file_bin))

input_blob = next(iter(net.inputs))
out_blobs = iter(net.outputs)
out_blob1 = next(out_blobs)
out_blob2 = next(out_blobs)
exec_net = plugin.load(network=net, num_requests=2)
input_dims = net.inputs[input_blob].shape
output_dim1 = net.outputs[out_blob1].shape
output_dim2 = net.outputs[out_blob2].shape
del net




in_frame = cv2.imread("/home/pi/Documents/model_project/test2.jpeg", cv2.IMREAD_COLOR)


frameWidth = in_frame.shape[1]
frameHeight = in_frame.shape[0]
	
	
#	inp = cv2.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
#                              (0, 0, 0), swapRB=False, crop=False)
#	net.setInput(inp)
#	out = net.forward()
    
	
    
n, c, h, w = input_dims
frame = cv2.resize(in_frame, (w, h))
frame = frame.transpose((2, 0, 1))
frame = frame.reshape((n,c,h,w))
exec_net.infer(inputs={input_blob:frame})
out1 = exec_net.requests[0].outputs[out_blob1]
out = exec_net.requests[0].outputs[out_blob2]



assert(len(BODY_PARTS) <= out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
	# Slice heatmap of corresponging body's part.
	heatMap = out[0, i, :, :]
	if i==0:
		print(heatMap)
		cv2.imshow('heatMap', heatMap)
		for j in range(32):
			for k in range(57):
				if heatMap[j, k] > 0.1:
					print(heatMap[j, k])
					print([j, k])
	if i==1:
		print(heatMap)
		cv2.imshow('heatMap2', heatMap)
		for j in range(32):
			for k in range(57):
				if heatMap[j, k] > 0.1:
					print(heatMap[j, k])
					print([j, k])
	#print(heatMap)
	# Originally, we try to find all the local maximums. To simplify a sample
	# we just find a global one. However only a single pose at the same time
	# could be detected this way.
	_, conf, _, point = cv2.minMaxLoc(heatMap)
	x = (frameWidth * point[0]) / out.shape[3]
	y = (frameHeight * point[1]) / out.shape[2]

	# Add a point if it's confidence is higher than threshold.
	points.append((int(x), int(y)) if conf > thr else None)

for pair in POSE_PAIRS:
	partFrom = pair[0]
	partTo = pair[1]
	assert(partFrom in BODY_PARTS)
	assert(partTo in BODY_PARTS)

	idFrom = BODY_PARTS[partFrom]
	idTo = BODY_PARTS[partTo]

	if points[idFrom] and points[idTo]:
		cv2.line(in_frame, points[idFrom], points[idTo], (0, 255, 0), 3)
		cv2.ellipse(in_frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
		cv2.ellipse(in_frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

	
#cv2.imshow('OpenPose using OpenCV', in_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

