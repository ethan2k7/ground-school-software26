import cv2
import numpy as np

VIDEO_NAME = "Minecraft_stitch_test.mp4"
FRAME_SKIP = 5
OUTPUT_NAME = "Minecraft_stitch_test.jpg"

#retrieves video from the file
video = cv2.VideoCapture(VIDEO_NAME)

#sets up list for frames
frames = []
frame_number = 0

#store the original video resolution
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Original resolution:", original_width, "x", original_height)

while True:

    success, frame = video.read()

    if not success:
        break

    # Only keep every 5th frame
    if frame_number % FRAME_SKIP == 0:

        # Make frame smaller
        width = 960
        height = int(frame.shape[0] * width / frame.shape[1])

        frame = cv2.resize(frame, (width, height))

        frames.append(frame)

        print("Saved frame:", frame_number)

    frame_number += 1


video.release()

orb = cv2.ORB_create(2000)
#matches features between images
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#starts first frame
# Create a large blank canvas
canvas_width = 960 * len(frames)
canvas_height = 1500

canvas = np.zeros(
    (canvas_height, canvas_width, 3),
    dtype=np.uint8
)

#setting frame position on canvas
x_offset = canvas_width // 2 - 480
y_offset = 200

canvas[
    y_offset:y_offset + frames[0].shape[0],
    x_offset:x_offset + frames[0].shape[1]
] = frames[0]

#position check
current_x = x_offset
current_y = y_offset

#stich frames
for i in range(1, len(frames)):

    print()
    print("Processing frame", i, "of", len(frames) - 1)

    previous_frame = frames[i - 1]
    current_frame = frames[i]


    #features in previous frame
    keypoints1, descriptors1 = orb.detectAndCompute(
        previous_frame,
        None
    )


    #features in current frame
    keypoints2, descriptors2 = orb.detectAndCompute(
        current_frame,
        None
    )

    #match features
    matches = matcher.match(
        descriptors1,
        descriptors2
    )


    #verify matching features
    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = matches[:50]


    print("Good matches:", len(good_matches))


    #need enough matches
    if len(good_matches) < 5:

        print("Not enough matches.")
        continue


    #calculate movement between frames

    movements = []

    for match in good_matches:

        point1 = keypoints1[
            match.queryIdx
        ].pt

        point2 = keypoints2[
            match.trainIdx
        ].pt

        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        movements.append((dx, dy))


    #find average movement
    average_dx = np.median(
        [movement[0] for movement in movements]
    )

    average_dy = np.median(
        [movement[1] for movement in movements]
    )


    print("Movement:", average_dx, average_dy)


    #adjust frame position on canvas along with camera movement
    current_x += int(average_dx)
    current_y += int(average_dy)


    #make sure frame stays inside canvas
    if current_x < 0:
        current_x = 0

    if current_y < 0:
        current_y = 0

    if current_x + current_frame.shape[1] > canvas_width:
        print("Reached right edge.")
        break

    if current_y + current_frame.shape[0] > canvas_height:
        print("Reached bottom edge.")
        break

    #add frames onto canvas
    canvas[
        current_y:current_y + current_frame.shape[0],
        current_x:current_x + current_frame.shape[1]
    ] = current_frame


#save final image
cv2.imwrite(
    OUTPUT_NAME,
    canvas
)

print()
print("DONE!")
print("Final image saved as:", OUTPUT_NAME)

#display image
cv2.imshow(
    "Final Flight Path",
    canvas
)

cv2.waitKey(0)
cv2.destroyAllWindows()