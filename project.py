import numpy as np
import micasense.metadata as metadata
import math

from pyproj import Proj, transform, CRS
from usng import LLtoUTM


# calibrated
omega = 6.06719009
phi = 3.73175178
kappa = 42.64839719

# from exif
roll = 8.06813
pitch = 27.133644
yaw = -28.643812


# Coordinates (latitude, longitude, altitude)
latitude = 47.5958150714778
longitude = 19.3715198189694
altitude = 293.9536051  # in meters

# pix4d uses UTM
utm = LLtoUTM(latitude, longitude)
print(utm)

# Camera intrinsic parameters from Pix4D calibration file
focal_length_px = 1661.2583368971314  # focal length in pixels
px_p = 611.8027223624858  # principal point x-coordinate in pixels
py_p = 467.06350970174316  # principal point y-coordinate in pixels

# Constructing the intrinsic matrix K
K = np.array([[focal_length_px, 0, px_p], [0, focal_length_px, py_p], [0, 0, 1]])

print("Intrinsic Matrix K:")


EARTH_RADIUS_METERS = 6371010
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F**2


def extract_gps(meta):
    latitude = meta.get_item("EXIF:GPSLatitude")
    latitude_ref = meta.get_item("EXIF:GPSLatitudeRef")
    longitude = meta.get_item("EXIF:GPSLongitude")
    longitude_ref = meta.get_item("EXIF:GPSLongitudeRef")
    altitude = meta.get_item("EXIF:GPSAltitude")

    print(latitude, latitude_ref, longitude, longitude_ref, altitude)

    if latitude and latitude_ref and longitude and longitude_ref:
        coords = lla2ecef(latitude, longitude, altitude, latitude_ref, longitude_ref)
        coords2 = transformLonLatAltToEcef(
            latitude, longitude, altitude, latitude_ref, longitude_ref
        )
        return coords, coords2


def transformLonLatAltToEcef(lat, lon, alt, lat_ref, lon_ref):
    """
    Transform tuple lon,lat,alt (WGS84 degrees, meters) to tuple ECEF
    x,y,z (meters)
    """

    lat = lat if lat_ref == "N" else -lat
    lon = lon if lon_ref == "E" else -lon

    print(lat)
    print(lon)

    a, e2 = WGS84_A, WGS84_E2

    lat = np.radians(lat)
    lon = np.radians(lon)
    chi = math.sqrt(1 - e2 * math.sin(lat) ** 2)
    q = (a / chi + alt) * math.cos(lat)
    return (
        q * math.cos(lon),
        q * math.sin(lon),
        ((a * (1 - e2) / chi) + alt) * math.sin(lat),
    )


# The conversion from WGS-84 to Cartesian has an analytical solution
def lla2ecef(lat, lon, alt, lat_ref, lon_ref):
    a = 6378137
    a_sq = a**2
    e = 8.181919084261345e-2
    e_sq = e**2
    b_sq = a_sq * (1 - e_sq)

    lat = lat if lat_ref == "N" else -lat
    lon = lon if lon_ref == "E" else -lon

    lat = np.radians(lat)
    lon = np.radians(lon)

    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e_sq) * N + alt) * np.sin(lat)

    return [x.item(), y.item(), z.item()]


def calculate_distance_ecef(point1, point2):
    return np.linalg.norm(point1 - point2)


def create_rotation_matrix(omega, phi, kappa):
    omega = np.radians(omega)  # Convert to radians
    phi = np.radians(phi)
    kappa = np.radians(kappa)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(omega), -np.sin(omega)],
            [0, np.sin(omega), np.cos(omega)],
        ]
    )

    Ry = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
    )

    Rz = np.array(
        [
            [np.cos(kappa), -np.sin(kappa), 0],
            [np.sin(kappa), np.cos(kappa), 0],
            [0, 0, 1],
        ]
    )

    R = Rz @ Ry @ Rx
    return R


# def project_pixel_to_3d(u, v, K, depth=0):
#     # Inverse of the intrinsic matrix
#     K_inv = np.linalg.inv(K)
#     # Convert pixel coordinates to homogeneous coordinates
#     pixel_homogeneous = np.array([u, v, 1])
#     # Apply the inverse camera matrix to get normalized camera coordinates
#     point_3d_normalized = np.dot(K_inv, pixel_homogeneous)
#     # Assume a depth (distance along the camera's z-axis)
#     point_3d = point_3d_normalized * depth
#     return point_3d
def ecef_to_lla(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137  # semi-major axis
    e = 8.1819190842622e-2  # eccentricity

    # calculations
    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return (lat, lon, alt)


# Example ECEF coordinates
x = 674362.493
y = 250228.501
z = 229.347

# Convert ECEF to LLA
latitude, longitude, altitude = ecef_to_lla(x, y, z)
print("Latitude:", latitude)
print("Longitude:", longitude)
print("Altitude:", altitude)


def project_pixel_to_3d(u, v, K, R, C, depth):
    # Inverse of the intrinsic matrix to get normalized camera coordinates
    K_inv = np.linalg.inv(K)
    # Convert pixel coordinates to homogeneous coordinates
    pixel_homogeneous = np.array([u, v, 1])
    # Camera coordinates
    cam_coords = np.dot(K_inv, pixel_homogeneous) * depth
    # Apply the rotation matrix and add the translation component (camera position)
    world_coords = np.dot(R, cam_coords) + C
    return world_coords


imageName = "gre.TIF"
meta = metadata.Metadata(imageName)
coords, coords2 = extract_gps(meta)
print(coords)
print(coords2)

# Extrinsic parameters derived from omega, phi, kappa (example values in degrees)
R = create_rotation_matrix(omega, phi, kappa)
gps_position = np.array(coords)
T = np.array([[0], [0], [gps_position[2]]])  # Camera's altitude in world coordinates

# Image point (example pixel coordinates)
image_point = (983, 328)  # gcp near cars


# ground_point1 = project_pixel_to_3d(image_point[0], image_point[1], K, 50)
# ground_point2 = project_pixel_to_3d(image_point[0], image_point[1], K, 0)

# print(coords)
# print(ground_point1)
# print(ground_point2)
# Project to ground
# ground_altitude = 50  # meters
# ground_point_ecef = pixel_to_ecef(image_point, K, R, coords, ground_altitude)
# print("Ground Point in ECEF Coordinates:", ground_point_ecef)
