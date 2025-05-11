from PIL import Image, ImageDraw
import ast
import json
import os
import argparse

# Default skeleton file path
DEFAULT_SKELETON_FILE = os.path.join(os.path.dirname(__file__), 'skeleton_points/boy64_skeleton.json')

# Image dimensions
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 1000
BACKGROUND_COLOR = "white"

# Point and line styling
POINT_RADIUS = 6
POINT_COLOR = "red"
LINE_COLOR = "blue"
LINE_WIDTH = 3

# Define connections between keypoints (pairs of labels)
# These are based on common human skeleton structures and the provided labels.
# "LEFT ARM" and "RIGHT ARM" are assumed to be wrists.
# "LEFT LEG" and "RIGHT LEG" are assumed to be ankles.
POSE_PAIRS = [
    # Head
    ("NOSE", "NECK"),
    ("NOSE", "LEFT EYE"),
    ("LEFT EYE", "LEFT EAR"),
    ("NOSE", "RIGHT EYE"),
    ("RIGHT EYE", "RIGHT EAR"),
    # Torso
    ("NECK", "LEFT SHOULDER"),
    ("NECK", "RIGHT SHOULDER"),
    ("LEFT SHOULDER", "LEFT HIP"),
    ("RIGHT SHOULDER", "RIGHT HIP"),
    # ("LEFT_HIP", "RIGHT_HIP"), # Optional: connect hips directly
    # Left Arm
    ("LEFT SHOULDER", "LEFT ELBOW"),
    ("LEFT ELBOW", "LEFT ARM"),  # LEFT ARM is wrist
    # Right Arm
    ("RIGHT SHOULDER", "RIGHT ELBOW"),
    ("RIGHT ELBOW", "RIGHT ARM"),  # RIGHT ARM is wrist
    # Left Leg
    ("LEFT HIP", "LEFT KNEE"),
    ("LEFT KNEE", "LEFT LEG"),  # LEFT LEG is ankle
    # Right Leg
    ("RIGHT HIP", "RIGHT KNEE"),
    ("RIGHT KNEE", "RIGHT LEG"),  # RIGHT LEG is ankle
]


def load_keypoints_from_file(json_file_path):
    """
    Loads keypoints from a JSON file.
    The file should contain a list of dictionaries, each representing a keypoint.
    """
    with open(json_file_path, 'r') as f:
        keypoints = json.load(f)
    return keypoints


def parse_keypoints(data_string):
    """
    Parses the string data into a list of dictionaries.
    Each dictionary represents a keypoint.
    For backward compatibility.
    """
    keypoints = []
    for line in data_string.strip().split("\n"):
        if line.strip():
            keypoint_dict = ast.literal_eval(line)
            keypoints.append(keypoint_dict)
    return keypoints


def draw_pose_on_image(keypoints_data, img_width, img_height):
    """
    Draws the pose (keypoints and skeleton) on a new blank image.
    """
    # Create a blank image
    image = Image.new("RGB", (img_width, img_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    # Store keypoints in a dictionary for easy lookup by label
    # Coordinates are normalized, so they need to be scaled to image dimensions
    points_map = {}
    for kp in keypoints_data:
        label = kp.get("label")
        # Normalize coordinates from 0-1 range to image pixel range
        x = kp.get("x") * img_width
        y = kp.get("y") * img_height
        if label and x is not None and y is not None:
            points_map[label] = (x, y)

    # --- Draw Skeleton Lines ---
    # Iterate through the defined pairs of connected keypoints
    for part_a_label, part_b_label in POSE_PAIRS:
        # Get coordinates for both parts of the pair
        part_a_coords = points_map.get(part_a_label)
        part_b_coords = points_map.get(part_b_label)
        
        # Draw a line if both keypoints exist
        if part_a_coords and part_b_coords:
            draw.line([part_a_coords, part_b_coords], fill=LINE_COLOR, width=LINE_WIDTH)

    # --- Draw Keypoints ---
    # Iterate through all detected keypoints and draw them on the image
    for label, (x, y) in points_map.items():
        # Define the bounding box for the ellipse (circle)
        # (x0, y0, x1, y1) where (x0,y0) is top-left and (x1,y1) is bottom-right
        bbox = [x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS]
        draw.ellipse(bbox, fill=POINT_COLOR, outline=POINT_COLOR)

        # Optional: Draw labels next to points
        # draw.text((x + POINT_RADIUS + 2, y - POINT_RADIUS), label, fill="black")

    return image


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize skeleton keypoints from a JSON file')
    parser.add_argument('--skeleton', type=str, default=DEFAULT_SKELETON_FILE,
                        help='Path to the skeleton JSON file')
    parser.add_argument('--width', type=int, default=IMAGE_WIDTH,
                        help='Width of the output image')
    parser.add_argument('--height', type=int, default=IMAGE_HEIGHT,
                        help='Height of the output image')
    parser.add_argument('--output', type=str, default="pose_visualization.png",
                        help='Output filename')
    args = parser.parse_args()
    
    # 1. Load the keypoints from the JSON file
    parsed_keypoints = load_keypoints_from_file(args.skeleton)
    
    # 2. Draw the pose onto a new image
    print(f"Loaded {len(parsed_keypoints)} keypoints from {args.skeleton}")
    pose_image = draw_pose_on_image(parsed_keypoints, args.width, args.height)

    # 3. Save the resulting image
    pose_image.save(args.output)
    print(f"Image saved as {args.output}")

    # 4. Display the image (optional)
    pose_image.show()
