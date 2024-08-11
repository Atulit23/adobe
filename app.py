from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from scipy.interpolate import interp1d
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app) 

CLOUDINARY_UPLOAD_URL = 'https://api.cloudinary.com/v1_1/ddvajyjou/image/upload'
UPLOAD_PRESET = 'nb6tvi1b'

CLOUDINARY_UPLOAD_URL = 'https://api.cloudinary.com/v1_1/ddvajyjou/image/upload'
UPLOAD_PRESET = 'nb6tvi1b'

# Euclidean distance function
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

# Shape regularization function
def regularize_shape(shape, c):
    c = c.squeeze()
    num_points = 60

    if shape == "rectangle" or shape == "square":
        x_sorted = np.sort(c[:, 0])
        y_sorted = np.sort(c[:, 1])

        c[0][0] = c[1][0] = x_sorted[0]
        c[2][0] = c[3][0] = x_sorted[-1]
        c[0][1] = c[3][1] = y_sorted[0]
        c[1][1] = c[2][1] = y_sorted[-1]

    elif shape == "circle" or shape == "ellipse":
        if shape == "circle":
            center, radius = cv2.minEnclosingCircle(c)
            circle = cv2.ellipse2Poly((int(center[0]), int(center[1])),
                                      (int(radius), int(radius)),
                                      0, 0, 360, 360 // num_points)
            return circle.reshape((-1, 1, 2))
        elif shape == "ellipse":
            ellipse = cv2.fitEllipse(c)
            ellipse_contour = cv2.ellipse2Poly((int(ellipse[0][0]),
                                                int(ellipse[0][1])),
                                               (int(ellipse[1][0]//2),
                                                int(ellipse[1][1]//2)),
                                               int(ellipse[2]), 0, 360, 360 // num_points)
            return ellipse_contour.reshape((-1, 1, 2))

    if not np.array_equal(c[0], c[-1]):
        c = np.append(c, [c[0]], axis=0)

    return c.reshape((-1, 1, 2))

# ShapeDetector class
class ShapeDetector:
    def __init__(self):
        pass
    
    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if cv2.isContourConvex(approx):
                shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
            else:
                shape = "rounded rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) >= 6:
            area = cv2.contourArea(c)
            circularity = (4 * np.pi * area) / (peri ** 2)
            if 0.8 <= circularity <= 1.2:
                if len(approx) > 8:
                    shape = "circle"
                else:
                    shape = "ellipse"
            else:
                if cv2.isContourConvex(approx):
                    shape = "regular polygon" if len(approx) < 10 else "star"
                else:
                    shape = "star"
                
        return shape, approx

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file, header=None)
        points = data.values[:, 2:]
        contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))

        distance_threshold = 10

        contours = []
        current_contour = [contour[0]]

        for i in range(1, len(contour)):
            if euclidean_dist(contour[i-1], contour[i]) > distance_threshold:
                contours.append(np.array(current_contour, dtype=np.float32))
                current_contour = [contour[i]]
            else:
                current_contour.append(contour[i])

        if current_contour:
            contours.append(np.array(current_contour, dtype=np.float32))

        shapes = []
        regularized_contours = []

        for c in contours:
            sd = ShapeDetector()
            shape, approx = sd.detect(c)
            reg_c = regularize_shape(shape, approx)
            shapes.append((shape, reg_c))
            regularized_contours.append(reg_c)

        # Plot and save the image
        plt.figure(figsize=(10, 10))
        for shape, reg_c in shapes:
            plt.plot(reg_c[:, 0, 0], reg_c[:, 0, 1], 'b-', label=f'Regularized {shape}')
        plt.title('Regularized Contours')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        
        img_path = 'contours.png'
        plt.savefig(img_path)
        plt.close()

        # Upload image to Cloudinary
        with open(img_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {'upload_preset': UPLOAD_PRESET}
            response = requests.post(CLOUDINARY_UPLOAD_URL, files=files, data=data)
            result = response.json()

        if response.status_code == 200:
            return jsonify({'url': result.get('secure_url')})
        else:
            return jsonify({'error': 'Failed to upload image'}), 500

    return jsonify({'error': 'Invalid file format'}), 400



def read_csv(file):
    np_path_XYs = np.genfromtxt(file, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def get_start_end_points(curves):
    start_end_points = []
    for curve in curves:
        tup = (curve[0], curve[-1])
        start_end_points.append(tup)  # (start_point, end_point)
    return start_end_points

def are_curves_joined(start_end_points, tolerance):
    num_curves = len(start_end_points)
    joined_pairs = []
    for i in range(num_curves):
        start_i, end_i = start_end_points[i]
        for j in range(num_curves):
            if i != j:
                start_j, end_j = start_end_points[j]
                if np.linalg.norm(end_i - start_j) < tolerance or np.linalg.norm(start_i - end_j) < tolerance:
                    joined_pairs.append((i, j))
    return joined_pairs

def merge_curves(curves, start_end_points, tolerance):
    joined_pairs = are_curves_joined(start_end_points, tolerance)
    
    merged_curves = []
    visited = set()
    
    def merge_curve_pair(curve_a, curve_b):
        return np.concatenate([curve_a, curve_b[1:]])
    
    def find_curve(i, all_curves):
        stack = [i]
        merged = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            curve = all_curves[current]
            merged.append(curve)
            for (a, b) in joined_pairs:
                if a == current and b not in visited:
                    stack.append(b)
                elif b == current and a not in visited:
                    stack.append(a)
        return merged
    
    all_curves = [curve for sublist in curves for curve in sublist]
    for i in range(len(all_curves)):
        if i not in visited:
            connected_curves = find_curve(i, all_curves)
            if connected_curves:
                merged_curve = connected_curves[0]
                for curve in connected_curves[1:]:
                    merged_curve = merge_curve_pair(merged_curve, curve)
                merged_curves.append(merged_curve)
    
    new_start_end_points = [(curve[0], curve[-1]) for curve in merged_curves]
    
    return merged_curves, new_start_end_points

@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        points = read_csv(file)
        
        start_end_points = []
        for path in points:
            start_end_points.extend(get_start_end_points(path))

        tolerance = 2.0
        merged_curves, new_start_end_points = merge_curves(points, start_end_points, tolerance)

        # Plot the merged curves
        plt.figure(figsize=(10, 10))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, curve in enumerate(merged_curves):
            c = colors[i % len(colors)]
            plt.plot(curve[:, 0], curve[:, 1], c=c, linewidth=2)
        
        plt.title('Merged Curves')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Upload image to Cloudinary
        files = {'file': buf}
        data = {'upload_preset': UPLOAD_PRESET}
        response = requests.post(CLOUDINARY_UPLOAD_URL, files=files, data=data)
        result = response.json()

        if response.status_code == 200:
            return jsonify({
                'url': result.get('secure_url'),
                'num_merged_curves': len(merged_curves)
            })
        else:
            return jsonify({'error': 'Failed to upload image'}), 500

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)