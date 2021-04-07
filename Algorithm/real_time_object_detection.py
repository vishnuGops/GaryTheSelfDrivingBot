# USAGE
# python real_time_object_detection.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --output output/massachusetts_output.avi --prototxt MobileNetSSD_deploy.prototxt.txt --model1 MobileNetSSD_deploy.caffemodel




# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
import numpy as np
import argparse
import imutils
import time
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c1", "--classes", required=True,
	help="path to .txt file containing class labels")

ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-s", "--show", type=int, default=1,
	help="whether or not to display frame to screen")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="desired width (in pixels) of input image")


ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m1", "--model1", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES1 = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

CLASSES2 = open(args["classes"]).read().strip().split("\n")


COLORS1 = np.random.uniform(0, 255, size=(len(CLASSES1), 3))

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")
else:
	# initialize a list of colors to represent each class label in
	# the mask (starting with 'black' for the background/unlabeled
	# regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model1"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

print("[INFO] loading model...")
net1 = cv2.dnn.readNet(args["model"])

# initialize the video stream and pointer to output video file
#vs = cv2.VideoCapture(args["video"])
writer = None







capture = cv2.VideoCapture(0)  #read the video
capture.set(3,320.0) #set the size
capture.set(4,240.0)  #set the size
capture.set(5,15)  #set the frame rate

# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(capture.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1


for i in range(0,2):
        flag, trash = capture.read() #starting unwanted null value

while cv2.waitKey(1) != 27:
        flag, frame = capture.read() #read the video in frames
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert each frame to grayscale.
        blur=cv2.GaussianBlur(gray,(5,5),0)#blur the grayscale image
        ret,th1 = cv2.threshold(blur,35,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#using threshold remave noise
        ret1,th2 = cv2.threshold(th1,127,255,cv2.THRESH_BINARY_INV)# invert the pixels of the image frame
        _ , contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find the contours
        cv2.imshow("Frame", frame)

        cv2.drawContours(frame,contours,-1,(0,255,0),3)
        #cv2.imshow('frame',frame) #show video
        for cnt in contours:
           if cnt is not None:
            area = cv2.contourArea(cnt)# find the area of contour
           if area>=500 :
            # find moment and centroid
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cx <= 150:
                l = (cx * 100 / 160)
                print("Left")
                #PWMR.start(0)
                #PWML.start(0)
                #PWMR1.ChangeDutyCycle(100)
                #PWML1.ChangeDutyCycle(abs(l - 25))
                #time.sleep(.08)

            elif cx >= 170:
                r = ((320 - cx) * 100 / 160)
                print("Right")
                #PWMR.start(0)
                #PWML.start(0)
                #PWMR1.ChangeDutyCycle(abs(r - 25))
                #PWML1.ChangeDutyCycle(100)
                #time.sleep(.08)

            elif cx > 151 and cx < 169:
                print("Straight")
                #PWMR.start(0)
                #PWML.start(0)
                #PWMR1.ChangeDutyCycle(96)
                #PWML1.ChangeDutyCycle(100)
                #time.sleep(.3)

            else:
                print("GOooo")





    # loop over the frames from the video stream
    #while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        #frame = capture.read()
        #frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES1[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS1[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS1[idx], 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

        # loop over frames from the video file stream
    #while True:
        # read the next frame from the file
        #(grabbed, frame) = vs.read()


        # construct a blob from the frame and perform a forward pass
        # using the segmentation model
        frame = imutils.resize(frame, width=args["width"])
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
                                     swapRB=True, crop=False)
        net1.setInput(blob)
        start = time.time()
        output = net1.forward()
        end = time.time()

        # infer the total number of classes along with the spatial
        # dimensions of the mask image via the shape of the output array
        (numClasses, height, width) = output.shape[1:4]

        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        classMap = np.argmax(output[0], axis=0)

        # given the class ID map, we can map each of the class IDs to its
        # corresponding color
        mask = COLORS[classMap]

        # resize the mask such that its dimensions match the original size
        # of the input frame
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input frame with the mask
        # to form an output visualization
        output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (output.shape[1], output.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(output)

        # check to see if we should display the output frame to our screen
        #if args["show"] > 0:
        cv2.imshow("Frame1", output)
        key = cv2.waitKey(1) & 0xFF



# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
writer.release()
capture.stop()