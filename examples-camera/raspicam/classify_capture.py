import argparse
import numpy as np
import os
import time
import cv2
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

def low_pass_filter(prev_pos, current_pos, alpha):
    """Apply a low-pass filter to smooth the airplane's location."""
    if prev_pos is None:
        return current_pos
    return (1 - alpha) * np.array(prev_pos) + alpha * np.array(current_pos)

def draw_airplane_marker(image, pos, confidence):
    """Draw the airplane marker with color based on the confidence score."""
    confidence = max(0.0, min(confidence, 1.0))  # Ensure confidence within [0, 1]
    
    red = int(255 * (1 - confidence))
    green = int(255 * confidence)
    color = (0, green, red)  # OpenCV uses BGR format

    if pos is not None:
        center = (int(pos[0]), int(pos[1]))
        radius = 20
        cv2.circle(image, center, radius, color, thickness=-1)
    else:
        print("No position provided to draw marker.")

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.06,
                        help='classifier score threshold')
    args = parser.parse_args()

    # Load labels
    with open(args.labels, 'r') as f:
        labels = {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1].strip() for line in f.readlines()}

    # Initialize interpreter
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Get model input details
    input_details = interpreter.get_input_details()
    _, height_NN_input, width_NN_input, _ = input_details[0]['shape']
    print(f"Expected input size: {width_NN_input} x {height_NN_input}")

    num_segments_x = 3
    num_segments_y = 2
    stream_width = 1920
    stream_height = 1080
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (stream_width, stream_height), "format": "BGR888"},
        raw=None
    )
    picam2.configure(camera_config)
    picam2.start()

    fps_counter = []
    grid_x, grid_y = 0, 0

    # Initialize variables for low-pass filter and zooming behavior
    prev_airplane_pos = None
    prev_confidence = 0.0
    airplane_detected = False
    zoom_levels = [300, 600, stream_width]  # 300x300, 600x600, full frame
    zoom_idx = 0
    frames_without_detection = 0  # Tracks missed detections
    max_frames_no_detection = 10  # Number of frames to allow before resetting to sweeping

    try:
        while True:
            start_time = time.time()

            # Capture image
            image = picam2.capture_array()
            image_height, image_width, _ = image.shape

            cropped_image = image  # Initialize cropped_image in case no detection is made

            if airplane_detected:
                zoom_size = zoom_levels[zoom_idx]
                if prev_airplane_pos is not None:
                    center_x, center_y = int(prev_airplane_pos[0]), int(prev_airplane_pos[1])

                    # Calculate boundaries for zoom
                    x_start = max(0, center_x - zoom_size // 2)
                    y_start = max(0, center_y - zoom_size // 2)
                    x_end = min(image_width, center_x + zoom_size // 2)
                    y_end = min(image_height, center_y + zoom_size // 2)

                    if x_end > x_start and y_end > y_start:
                        cropped_image = image[y_start:y_end, x_start:x_end]

                        # Resize image to NN input size
                        crop_resized_image = cv2.resize(cropped_image, (width_NN_input, height_NN_input), interpolation=cv2.INTER_LINEAR)
                        crop_rgb_image = cv2.cvtColor(crop_resized_image, cv2.COLOR_BGR2RGB)
                        crop_input_data = np.expand_dims(crop_rgb_image, axis=0)

                        common.set_input(interpreter, crop_input_data)
                        interpreter.invoke()
                        detections = detect.get_objects(interpreter, score_threshold=args.threshold)

                        if len(detections) == 0:
                            frames_without_detection += 1
                        else:
                            frames_without_detection = 0

                        for detection in detections:
                            if detection.id == 4:
                                xmin = int(detection.bbox.xmin * zoom_size / width_NN_input) + x_start
                                ymin = int(detection.bbox.ymin * zoom_size / height_NN_input) + y_start
                                xmax = int(detection.bbox.xmax * zoom_size / width_NN_input) + x_start
                                ymax = int(detection.bbox.ymax * zoom_size / height_NN_input) + y_start

                                current_airplane_pos = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                                current_confidence = detection.score
                                prev_airplane_pos = low_pass_filter(prev_airplane_pos, current_airplane_pos, current_confidence)
                                prev_confidence = current_confidence

                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                label = f'{labels.get(detection.id, "Unknown")}: {detection.score:.2f}'
                                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(image, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin), (0, 255, 0), cv2.FILLED)
                                cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                                print(f"Zoom {zoom_size} - Detected airplane at ({prev_airplane_pos[0]:.2f}, {prev_airplane_pos[1]:.2f})")
                                break
                        zoom_idx = (zoom_idx + 1) % len(zoom_levels)

                    if frames_without_detection > max_frames_no_detection:
                        airplane_detected = False
                        frames_without_detection = 0
                else:
                    print("Previous airplane position is None.")
            else:
                segment_width = image_width // num_segments_x
                segment_height = image_height // num_segments_y

                x_start = grid_x * segment_width
                y_start = grid_y * segment_height
                x_end = x_start + segment_width
                y_end = y_start + segment_height

                if x_end <= image_width and y_end <= image_height:
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

                    cropped_image = image[y_start:y_end, x_start:x_end]

                    if cropped_image is not None:
                        crop_resized_image = cv2.resize(cropped_image, (width_NN_input, height_NN_input), interpolation=cv2.INTER_LINEAR)
                        crop_rgb_image = cv2.cvtColor(crop_resized_image, cv2.COLOR_BGR2RGB)
                        crop_input_data = np.expand_dims(crop_rgb_image, axis=0)

                        common.set_input(interpreter, crop_input_data)
                        interpreter.invoke()
                        detections = detect.get_objects(interpreter, score_threshold=args.threshold)

                        if len(detections) == 0:
                            print("No detections in this sweep segment.")
                        else:
                            for detection in detections:
                                if detection.id == 4:
                                    airplane_detected = True
                                    xmin = int(detection.bbox.xmin * segment_width / width_NN_input) + x_start
                                    ymin = int(detection.bbox.ymin * segment_height / height_NN_input) + y_start
                                    xmax = int(detection.bbox.xmax * segment_width / width_NN_input) + x_start
                                    ymax = int(detection.bbox.ymax * segment_height / height_NN_input) + y_start

                                    current_airplane_pos = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                                    current_confidence = detection.score
                                    prev_airplane_pos = low_pass_filter(prev_airplane_pos, current_airplane_pos, current_confidence)
                                    prev_confidence = current_confidence

                                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    label = f'{labels.get(detection.id, "Unknown")}: {detection.score:.2f}'
                                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(image, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin), (0, 255, 0), cv2.FILLED)
                                    cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                    break

                grid_x = (grid_x + 1) % num_segments_x
                if grid_x == 0:
                    grid_y = (grid_y + 1) % num_segments_y

            if prev_airplane_pos is not None:
                draw_airplane_marker(image, prev_airplane_pos, prev_confidence)

            fps_counter.append(time.time())
            if len(fps_counter) > 30:
                fps_counter = fps_counter[1:]
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0]) if len(fps_counter) > 1 else 0

            fps_text = 'FPS: {:.2f}'.format(fps)
            cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            display_image = cv2.resize(image, (300, int(image_height * (300 / image_width))), interpolation=cv2.INTER_AREA)
            zoom_display_image = cv2.resize(cropped_image, (300, 300), interpolation=cv2.INTER_AREA)

            cv2.imshow('Full Field of View', display_image)
            cv2.imshow('Zoomed-in View', zoom_display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
