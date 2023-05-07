import cv2
import numpy as np
from matplotlib import pyplot as plt

clicks = 0

def select_point(event, x, y, flags, param):
    global point_selected, top, bottom, clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if (clicks == 0):
            top = (x, y)
        if (clicks == 1):
            bottom = (x, y)
            point_selected = True
            cv2.destroyAllWindows()
        clicks += 1

def show_timer(time_elapsed):
    timer_window = np.zeros((200, 400), dtype=np.uint8)
    text = f"{int(time_elapsed // 60):02d}:{int(time_elapsed % 60):02d},{int((time_elapsed % 1) * 100):02d}"
    
    # Adjust the font scale based on the window size
    window_height, window_width = timer_window.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 5

    while True:
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        if text_width < window_width * 0.9 and text_height < window_height * 0.9:
            font_scale += 0.1
        else:
            font_scale -= 0.1
            break

    # Calculate the position of the text so it's centered
    x = (window_width - text_width) // 2
    y = (window_height + text_height) // 2

    cv2.putText(timer_window, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imshow('Timer', timer_window)


def filter_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_bgr = np.uint8([[[76, 83, 212]]])  # BGR values for the pink color #D4534C
    pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)
    hue = pink_hsv[0][0]
    lower_bound = np.array([0, hue[1] - 40, hue[2] - 40])
    upper_bound = np.array([360, hue[1] + 40, hue[2] + 40])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

def removeEnd(theo):
    # Find the index where the data begins to increase
    diff = np.diff(theo)
    increasing_index = np.where(diff > 0)[0]
    if increasing_index.size > 0:
        increasing_index = increasing_index[0] + 1
        # Set values to 0 after the index
        theo[increasing_index:] = 0
    return theo

def find_top_non_black_point(image, y_start):
    non_black_indices = np.argwhere(np.any(image[y_start:], axis=-1))
    if non_black_indices.size > 0:
        return non_black_indices[0][::-1] + [0, y_start]
    return None

def track_point(filename, totalHeight):
    global point_selected, top, bottom, clicks
    point_selected = False
    top = ()
    bottom = ()
    heights = []
    oldY = 0

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    if not ret:
        print("Error reading video file.")
        return

    frame_resized = cv2.resize(frame, (720, 720))
    filtered_frame = filter_colors(frame_resized)

    cv2.namedWindow('Point tracking selector')
    cv2.setMouseCallback('Point tracking selector', select_point)

    while not point_selected:
        frame_with_text = frame_resized.copy()
        if (clicks == 0):
            cv2.putText(frame_with_text, 'Select Top of Bottle', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Point tracking selector', frame_with_text)
        if (clicks == 1):
            cv2.putText(frame_with_text, 'Select Bottom of Bottle', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Point tracking selector', frame_with_text)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not point_selected:
        print("No point selected.")
        return

    cap = cv2.VideoCapture(filename)

    top = np.array([[[top[0], top[1]]]], dtype=np.float32)
    bottom = np.array([[[bottom[0], bottom[1]]]], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (720, 720))
        filtered_frame = filter_colors(frame_resized)

        yT = top.ravel()[1]
        cv2.line(frame_resized, (0, int(yT)), (720, int(yT)), (7, 217, 237), 5)

        yB = bottom.ravel()[1]
        cv2.line(frame_resized, (0, int(yB)), (720, int(yB)), (7, 217, 237), 5)

        y = find_top_non_black_point(filtered_frame,int(yT) + 1)
        if y is None: 
            y = oldY
        else:
            y = y[1]
        oldY = y
        cv2.line(frame_resized, (0, int(y)), (720, int(y)), (140, 241, 249), 5)

        height = yT - yB
        currentHeight = (float(y - yB) / height) * float(totalHeight)

        heights.append(currentHeight)

        cv2.imshow('Tracked Point', frame_resized)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        show_timer(current_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return heights

if __name__ == '__main__':
    totalHeight = 16.5 # [cm] / Height of the container
    g = 9.81
    r = 0.045 # [m] / Radius of the container
    A_c = np.pi * r**2 # [m²] / Area of water surface

    filename = input("Enter the video file path: ")
    hole = input("Enter the hole diameter in cm: ")
    heights = track_point(filename, totalHeight)
    heights = np.array(heights) * 0.01 # cm

    A_h = np.pi * ((float(hole) * 0.01)/2)**2 # [m²] / Area of hole

    time = np.arange(len(heights)) # Time in frames
    time = time / 30 # Time in seconds

    title = input("Enter a title for the graphs: ")
    
#    dpi = 96  # Display's DPI
#    width_in_inches = 1040 / dpi
#    height_in_inches = 480 / dpi

#    plt.figure(1, figsize=(width_in_inches, height_in_inches))
    plt.figure(1)
    plt.suptitle(title)
    # HEIGHT
    h_theo = ((A_h/A_c) * np.sqrt(g/2) * time - np.sqrt(heights[0]))**2
    h_theo = removeEnd(h_theo)

    plt.subplot(1, 2, 1)
    plt.plot(time, heights)
    plt.plot(time, h_theo)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.title('Height v. Time')
    plt.legend(["Experimental value", "Theoretical value"])

    # VELOCITY
    velocity = np.sqrt(2*g*heights)
    v_theo = np.sqrt(2*g*h_theo)
    v_theo = removeEnd(v_theo)

    plt.subplot(1, 2, 2)
    plt.plot(time, velocity)
    plt.plot(time, v_theo)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity v. Time')
    plt.legend(["Experimental value", "Theoretical value"])

    plt.show()