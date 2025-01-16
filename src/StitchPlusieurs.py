import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import imageio
import os
import re
from typing import List, Tuple, Optional
warnings.filterwarnings('ignore')

class PanoramaStitcher:
    def __init__(self, feature_method: str = 'sift', max_size: int = 1024):
        """
        Initialize the panorama stitcher.
        Args:
            feature_method: Feature extraction method ('sift', 'surf', 'orb', 'brisk')
            max_size: Maximum size for the longest edge of input images
        """
        self.feature_method = feature_method
        self.max_size = max_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        """
        height, width = image.shape[:2]
        if max(height, width) > self.max_size:
            scale = self.max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def read_and_convert_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read an image, resize it, and convert it to RGB and grayscale
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read the image at '{image_path}'")
        
        # Resize image before processing
        img = self.resize_image(img)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_rgb, img_gray

    def get_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        if self.feature_method == 'sift':
            descriptor = cv2.SIFT_create()
        elif self.feature_method == 'surf':
            descriptor = cv2.SURF_create()
        elif self.feature_method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif self.feature_method == 'orb':
            descriptor = cv2.ORB_create()
        else:
            raise ValueError("Unsupported feature method")
            
        keypoints, features = descriptor.detectAndCompute(image, None)
        return keypoints, features

    def match_features(self, features1: np.ndarray, features2: np.ndarray, ratio: float = 0.75) -> List:
        if self.feature_method in ['sift', 'surf']:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
        raw_matches = bf.knnMatch(features1, features2, k=2)
        matches = []
        
        for m, n in raw_matches:
            if m.distance < n.distance * ratio:
                matches.append(m)
                
        return matches

    def calculate_homography(self, keypoints1: List, keypoints2: List, matches: List, reproj_thresh: float = 4.0) -> Optional[Tuple]:
        if len(matches) < 4:
            return None

        keypoints1 = np.float32([kp.pt for kp in keypoints1])
        keypoints2 = np.float32([kp.pt for kp in keypoints2])

        points1 = np.float32([keypoints1[m.queryIdx] for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx] for m in matches])

        H, status = cv2.findHomography(points1, points2, cv2.RANSAC, reproj_thresh)
        return (H, status)

    def stitch_two_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Stitch two images together
        """
        # Get grayscale images
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Get features and matches
        kp1, feat1 = self.get_features(gray1)
        kp2, feat2 = self.get_features(gray2)
        matches = self.match_features(feat1, feat2)
        
        if len(matches) < 4:
            raise ValueError("Not enough matches found between images")
            
        # Calculate homography matrix
        result = self.calculate_homography(kp1, kp2, matches)
        if result is None:
            raise ValueError("Could not calculate homography between images")
            
        H, _ = result
        
        # Calculate dimensions of combined image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate the size of the combined image
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transform corners of first image
        corners1_transform = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_transform), axis=0)
        
        # Get the bounds
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel())
        
        # Translation matrix
        translation = np.array([[1, 0, -x_min],
                              [0, 1, -y_min],
                              [0, 0, 1]])
        
        # Warp first image
        output_shape = (x_max - x_min, y_max - y_min)
        warp_matrix = translation.dot(H)
        img1_warped = cv2.warpPerspective(img1, warp_matrix, output_shape)
        
        # Create output canvas
        output = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
        
        # Copy second image
        offset = [-x_min, -y_min]
        output[max(0, offset[1]):min(offset[1] + h2, output_shape[1]),
               max(0, offset[0]):min(offset[0] + w2, output_shape[0])] = img2
        
        # Blend the warped image
        mask = img1_warped != 0
        output[mask] = img1_warped[mask]
        
        return output

    def stitch_images(self, image_paths: List[str], output_path: str = "./panorama.jpeg") -> np.ndarray:
        """
        Stitch multiple images together in sequence
        """
        if len(image_paths) < 2:
            raise ValueError("At least two images are required for stitching")

        # Read all images
        print("Reading and resizing images...")
        images = []
        for path in image_paths:
            img, _ = self.read_and_convert_image(path)
            images.append(img)
            print(f"Processed {path} - Size: {img.shape[1]}x{img.shape[0]}")

        # Start with first image
        result = images[0]
        
        # Stitch each subsequent image
        for i in range(1, len(images)):
            print(f"Stitching image {i+1}/{len(images)}...")
            try:
                # Clear some memory
                if i > 1:
                    images[i-2] = None
                
                result = self.stitch_two_images(result, images[i])
                
                # Optionally resize the intermediate result if it gets too large
                if max(result.shape[0], result.shape[1]) > self.max_size * 2:
                    scale = (self.max_size * 2) / max(result.shape[0], result.shape[1])
                    new_size = (int(result.shape[1] * scale), int(result.shape[0] * scale))
                    result = cv2.resize(result, new_size, interpolation=cv2.INTER_AREA)
                    print(f"Resized intermediate result to {result.shape[1]}x{result.shape[0]}")
                
            except Exception as e:
                print(f"Error while stitching image {i+1}: {str(e)}")
                return None

        # Final resize if needed
        if max(result.shape[0], result.shape[1]) > self.max_size * 3:
            scale = (self.max_size * 3) / max(result.shape[0], result.shape[1])
            new_size = (int(result.shape[1] * scale), int(result.shape[0] * scale))
            result = cv2.resize(result, new_size, interpolation=cv2.INTER_AREA)
            print(f"Final result size: {result.shape[1]}x{result.shape[0]}")

        # Save and display result
        plt.figure(figsize=(20,10))
        plt.imshow(result)
        plt.axis('off')
        imageio.imwrite(output_path, result)
        plt.show()
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the stitcher with a maximum image size
    stitcher = PanoramaStitcher(feature_method='sift', max_size=1024)  # Vous pouvez ajuster cette valeur
    
    # Get all images from the img folder
    img_folder = "imgLerins"
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    def natural_sort_key(file_name):
        """
        Extract numbers from file names for natural sorting.
        Ensures numerical parts are treated as integers for correct ordering.
        """
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file_name)]


    # Sort image paths numerically
    image_paths = [
        os.path.join(img_folder, f) 
        for f in sorted(os.listdir(img_folder), key=natural_sort_key) 
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_paths:
        print(f"No images found in {img_folder} folder")
        exit()
        
    print(f"Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")
    
    # Create the panorama
    panorama = stitcher.stitch_images(image_paths, output_path="./output/panorama.jpeg")