import math

# Input data
altitude_h = 50.0  # relative altitude of the drone in meters
focal_length = 24.0  # camera focal length in mm
horizontal_fov = 84.0  # horizontal field of view of the camera in degrees
sensor_width = 6.3  # sensor width in mm (6.3mm for the Mavic Air 2)
horizontal_resolution = 4000  # horizontal image resolution in pixels
vertical_resolution = 3000  # vertical image resolution in pixels

# Data for the area to cover:
area_width = 2800  # width of the area to cover in meters
area_height = 1050  # height of the area to cover in meters

def calculate_drone_coverage_with_focal_length(focal_length: float, sensor_width: float, altitude_h: float, horizontal_resolution: int, vertical_resolution: int, horizontal_fov: float):

    # Calculate the vertical field of view based on the 4:3 image ratio
    vertical_fov = horizontal_fov / (4 / 3)  # Approximately 63.2° for the Mavic Air 2

    # Convert FOV angles to radians
    horizontal_fov_rad = math.radians(horizontal_fov / 2)
    vertical_fov_rad = math.radians(vertical_fov / 2)

    # Calculate the ground coverage width and height
    ground_width = 2 * altitude_h * math.tan(horizontal_fov_rad)
    ground_height = 2 * altitude_h * math.tan(vertical_fov_rad)

    # Calculate the ground coverage area
    ground_area_per_image = ground_width * ground_height

    # Calculate the ground sampling distance (GSD)
    gsd_horizontal = ground_width / horizontal_resolution
    gsd_vertical = ground_height / vertical_resolution

    # Display the results
    print("\nResults of coverage with focal length and sensor width taken into account:")
    print(f"Horizontal field of view (FOV): {horizontal_fov:.2f}°")
    print(f"Vertical field of view (FOV): {vertical_fov:.2f}°")
    print(f"Width of ground coverage area: {ground_width:.2f} meters")
    print(f"Height of ground coverage area: {ground_height:.2f} meters")
    print(f"Ground coverage area: {ground_area_per_image:.2f} square meters")
    print(f"Ground sampling distance (GSD) horizontal: {gsd_horizontal:.4f} meters/pixel")
    print(f"Ground sampling distance (GSD) vertical: {gsd_vertical:.4f} meters/pixel\n")
    return ground_width, ground_height, ground_area_per_image, gsd_horizontal, gsd_vertical

ground_width, ground_height, ground_area_per_image, gsd_horizontal, gsd_vertical = calculate_drone_coverage_with_focal_length(focal_length, sensor_width, altitude_h, horizontal_resolution, vertical_resolution, horizontal_fov)

def calculate_required_images_for_area():
    total_area = area_width * area_height
    print(f"Total area to cover: {total_area} square meters")
    required_full_images = total_area / ground_area_per_image
    print(f"Number of full images needed to cover the area: {math.ceil(required_full_images)}")
    return math.ceil(required_full_images)

number_of_images = calculate_required_images_for_area()
