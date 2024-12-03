import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math

# Load the YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = YOLO('yolov5n.pt')  # load an official model
model = YOLO('yolo11n.pt')  # load an official model
model.to(device)
yolo_classes = model.names


# Load the video
video = cv2.VideoCapture('/home/bharath/Downloads/test_codes/yolo/videos/test4.mp4')
output_filename = 'output_video2.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))



# track = True
track = False



def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    original_frame = background.copy()
    foreground = cv2.resize(foreground, (objSize, objSize))


    # Get the shape of the foreground image
    rows, cols, channels = foreground.shape

    # Calculate the center of the foreground image
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    # Rotate the foreground image
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    foreground = cv2.warpAffine(foreground, M, (cols, rows))

    # Overlay the rotated foreground image onto the background image
    for row in range(rows):
        for col in range(cols):
            if x + row < background.shape[0] and y + col < background.shape[1]:
                alpha = foreground[row, col, 3] / 255.0
                background[x + row, y + col] = alpha * foreground[row, col, :3] + (1 - alpha) * background[x + row, y + col]

    # Blend the foreground and background ROI using cv2.addWeighted
    result = background

    return result


def simulate_object(background, object_class, x, y):
    # Load the object image based on the class
    object_img = cv2.imread(f'assets/{object_class}.png', cv2.IMREAD_UNCHANGED)
    if object_img is None:
        return background
    # Simulate the object by overlaying it onto the background image
    # object_img = cv2.resize(object_img, (100, 100))
    background[y:y+100, x:x+100] = overlay_transparent(background[y:y+100, x:x+100], object_img, 0, 0, 0)

    return background



def add_myCar_overlay(background):
    overlay_img = cv2.imread('assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    # Get the shape of the overlay image
    rows, cols, _ = overlay_img.shape
    x = 550
    y = background.shape[0] - 200

    # Overlay the image onto the background
    overlay_img = overlay_transparent(background[y:y+rows, x:x+cols], overlay_img, 0, 0, 0, objSize=250)
    background[y:y+rows, x:x+cols] = overlay_img

    return background


def plot_object_bev(transformed_image_with_centroids, src_points ,dst_points , objs_):

  M = cv2.getPerspectiveTransform(src_points, dst_points)
  persObjs = []
  ## mark objs and ids
  for obj_ in objs_:
    if obj_:
      # Create a numpy array of the centroid coordinates
      centroid_coords = np.array([list(obj_[0])], dtype=np.float32)

      # Apply the perspective transformation to the centroid coordinates
      transformed_coords = cv2.perspectiveTransform(centroid_coords.reshape(-1, 1, 2), M)
      transformed_coords_ = tuple(transformed_coords[0][0].astype(int))

      # Draw a circle at the transformed centroid location
      cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=3, color=(0, 255, 0), thickness=-1)
      cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=12, color=(255, 255, 255), thickness=1)
      class_text = f"Class: {obj_[1]}"
      cv2.putText(transformed_image_with_centroids, class_text, (transformed_coords_[0] + 10, transformed_coords_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      persObjs.append([transformed_coords_, obj_[1]])

  return transformed_image_with_centroids, persObjs


frame_count = 0
centroid_prev_frame = []
tracking_objects = {}
tracking_id = 0


# Process each frame of the video
while True:
    # Read the next frame
    success, frame = video.read()
    if not success:
        break
    frame = cv2.resize(frame, (width, height))
    frame_count += 1
    image_ = np.zeros((height, width, 3), dtype=np.uint8)
    simulated_image = image_.copy()
    transformed_image_with_centroids = image_.copy()
    transformed_image_to_sim = image_.copy()
    simObjs = image_.copy()

    # Perform object detection on the frame

    results = model.track(frame, verbose=False, device=device)
    for predictions in results:
        if predictions is None:
            continue

        # Continue only if boxes and their ids are available
        if predictions.boxes is None or predictions.boxes.id is None:
            continue

        objs = []
        centroid_curr_frame = []

        #####################
        ##  OBJ DETECTION  ##
        #####################
        detections = predictions.boxes
        # If masks are present, iterate through both bbox and masks
        if predictions.boxes is not None:
            for bbox in predictions.boxes:
                for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                    xmin    = bbox_coords[0]
                    ymin    = bbox_coords[1]
                    xmax    = bbox_coords[2]
                    ymax    = bbox_coords[3]
                    centroid_x = int(xmin + xmax) // 2
                    centroid_y =  int(ymin + ymax) // 2

                    # Draw rectangle for the bounding box
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)

                    if int(classes) in [0, 1, 2, 3, 5, 7] and scores >= 0.3:
                        # Draw bounding box on the frame
                        color = (0, 0, 255)
                        object_label = f"{classes}: {scores:.2f}"
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                        cv2.putText(frame, object_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                        centroid_curr_frame.append([(centroid_x, centroid_y), yolo_classes[int(classes)]])
                        if track:
                            objs.append([(centroid_x, centroid_y), yolo_classes[int(classes)]])


    #####################
    ## OBJECT TRACKING ##
    #####################
    if track:
        if frame_count <= 2:
            for pt1, class_id in centroid_curr_frame:
                for pt2, class_id in centroid_prev_frame:
                    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                    if dist < 50:
                        tracking_objects[tracking_id] = pt1, class_id
                        tracking_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            for obj_id, pt2 in tracking_objects_copy.items():
                objects_exists = False
                for pt1, class_id in centroid_curr_frame:
                    dist = math.hypot(pt2[0][0] - pt1[0], pt2[0][1] - pt1[1])
                    if dist < 20:
                        tracking_objects[obj_id] = pt1, class_id
                        objects_exists = True
                        continue
                if not objects_exists:
                    tracking_objects.pop(obj_id)


        for obj_id, pt1 in tracking_objects.items():
            cv2.circle(frame, pt1[0], 3, (0, 255, 255), -1)
            # cv2.putText(frame, str(obj_id)+' '+str(pt1[1]), pt1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            if track:
                objs.append([pt1[0], pt1[1]])

        centroid_prev_frame = centroid_curr_frame.copy()


    #####################
    ##        BEV      ##
    #####################
    # Define the source points (region of interest) in the original image
    x1, y1 = 10, 720  # Top-left point
    x2, y2 = 530, 400  # Top-right point
    x3, y3 = 840, 400  # Bottom-right point
    x4, y4 = 1270, 720  # Bottom-left point
    src_points = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    # Draw the source points on the image (in red)
    # cv2.polylines(frame, [src_points.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

    # # Define the destination points (desired output perspective)
    u1, v1 = 370, 720  # Top-left point
    u2, v2 = 0+150, 0  # Top-right point
    u3, v3 = 1280-150, 0  # Bottom-right point
    u4, v4 = 900, 720  # Bottom-left point
    dst_points = np.float32([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])
    # # Draw the destination points on the image (in blue)
    # cv2.polylines(frame, [dst_points.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)

    # perspectivs plot and objs
    transformed_image_with_centroids, persObjs_ = plot_object_bev(transformed_image_with_centroids, src_points ,dst_points , objs)

    ### plot objs overlays
    for persObj_ in persObjs_:
        simObjs = simulate_object(transformed_image_to_sim, persObj_[1], persObj_[0][0], persObj_[0][1])
    # Add the car_img overlay to the simulated image
    simulated_image = add_myCar_overlay(simObjs)


    videoOut.write(simulated_image)
    # Display the simulated image and frame
    cv2.imshow("Video", frame)
    if track:
        cv2.imshow("Simulated Objects", simulated_image)
        cv2.imshow('Transformed Frame', transformed_image_with_centroids)
    # cv2.imwrite('test.jpg', simulated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
videoOut.release()
cv2.destroyAllWindows()
