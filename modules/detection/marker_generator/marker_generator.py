import cv2
import numpy as np

# === CHOOSE MODE HERE ===
#   Set 'mode' to one of: "aruco", "gridboard", or "charucoboard"
mode = "charucoboard"
# =========================


# ----------------------------------------
# Common dictionary (4x4_50) for all boards
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# ----------------------------------------


# ---------------------------
# PARAMETERS FOR "aruco" MODE
# ---------------------------
marker_id     = 8           # ID of the single ArUco marker to generate
marker_size   = 400         # Size in pixels (width=height) of the marker image
marker_output = "monocular_obstacle_avoidance/modules/detection/marker_generator/aruco.png"  # Output filename for the single marker


# -------------------------------
# PARAMETERS FOR "gridboard" MODE
# -------------------------------
markers_x        = 6        # Number of markers in X direction
markers_y        = 6        # Number of markers in Y direction
marker_length    = 0.11     # Marker side length in meters
marker_separation= 0.03     # Separation between markers in meters
grid_img_width   = 400      # Width of the output image in pixels
grid_img_height  = 400      # Height of the output image in pixels
grid_output      = "monocular_obstacle_avoidance/modules/detection/marker_generator/aruco_board.png"  # Output filename for the GridBoard


# ----------------------------------
# PARAMETERS FOR "charucoboard" MODE
# ----------------------------------
squares_x       = 6         # Number of ChArUco squares in X direction
squares_y       = 6         # Number of ChArUco squares in Y direction
square_length   = 0.018     # Side length of each ChArUco square in meters
marker_length_c = 0.014     # Side length of each ArUco marker inside the square (in meters)
charuco_px      = 640       # Total width (in pixels) of the ChArUco image
margin_px       = 20        # Margin (in pixels) around the Charuco board
charuco_output  = "monocular_obstacle_avoidance/modules/detection/marker_generator/charucoboard.png"  # Output filename for the ChArUco board


# ----------------------------------------
# FUNCTIONS TO GENERATE EACH TYPE OF BOARD
# ----------------------------------------

def generate_single_aruco(dictionary, marker_id, size, output_path):
    """
    Generate and save a single ArUco marker image.
    """
    # Generate the marker (grayscale)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    
    # Save to disk
    cv2.imwrite(output_path, marker_img)
    print(f"[INFO] Saved single marker (ID={marker_id}) to '{output_path}'")
    
    # Display window
    cv2.imshow(f"ArUco Marker ID {marker_id}", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gridboard(dictionary, mx, my, m_len, m_sep, img_w, img_h, output_path):
    """
    Generate and save an ArUco GridBoard image.
    """
    # Create GridBoard object
    board = cv2.aruco.GridBoard(
        (mx, my),
        markerLength=m_len,
        markerSeparation=m_sep,
        dictionary=dictionary
    )
    
    # Draw the board into an image
    board_img = board.generateImage((img_w, img_h), None, 1, 1)
    
    # Save to disk
    cv2.imwrite(output_path, board_img)
    print(f"[INFO] Saved ArUco GridBoard ({mx}×{my}) to '{output_path}'")
    
    # Display window
    cv2.imshow("Aruco GridBoard", board_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_charucoboard(dictionary, sx, sy, sq_len, mk_len, img_px, margin, output_path):
    """
    Generate and save a ChArUco board image.
    """
    # Create CharucoBoard object
    charuco = cv2.aruco.CharucoBoard(
        (sx, sy),
        sq_len,
        mk_len,
        dictionary
    )
    
    # Compute height so that aspect ratio = sy / sx
    aspect_ratio = sy / sx
    img_h = int(img_px * aspect_ratio)
    
    # Generate the Charuco board image
    charuco_img = cv2.aruco.CharucoBoard.generateImage(
        charuco,
        (img_px, img_h),
        marginSize=margin
    )
    
    # Save to disk
    cv2.imwrite(output_path, charuco_img)
    print(f"[INFO] Saved ChArUco board ({sx}×{sy}) to '{output_path}'")
    
    # Display window
    cv2.imshow("ChArUco Board", charuco_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------
# MAIN CONTROL BLOCK
# -------------------
if __name__ == "__main__":
    if mode == "aruco":
        generate_single_aruco(
            aruco_dict,
            marker_id,
            marker_size,
            marker_output
        )
    
    elif mode == "gridboard":
        generate_gridboard(
            aruco_dict,
            markers_x,
            markers_y,
            marker_length,
            marker_separation,
            grid_img_width,
            grid_img_height,
            grid_output
        )
    
    elif mode == "charucoboard":
        generate_charucoboard(
            aruco_dict,
            squares_x,
            squares_y,
            square_length,
            marker_length_c,
            charuco_px,
            margin_px,
            charuco_output
        )
    
    else:
        raise ValueError("Invalid mode! Choose 'aruco', 'gridboard', or 'charucoboard' at the top of the script.")
