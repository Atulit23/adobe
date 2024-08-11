import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(XYs, colors=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    if colors is None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i, XY in enumerate(XYs):
        c = colors[i % len(colors)]
        ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    ax.set_aspect('equal')
    plt.show()

def get_start_end_points(curves):
    start_end_points = []
    for curve in curves:
        tup = (curve[0], curve[-1])
        start_end_points.append(tup)
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

def draw(XYs, canvas_size=(800, 800)):
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    margin = 30
    multi = 3
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)]
    
    for i, XY in enumerate(XYs):
        color = colours[i % len(colours)]
        for point in XY:
            # Draw each point as a circle
            x, y = int(margin + point[0] * multi), int(margin + point[1] * multi)
            cv2.circle(canvas, (x, y), radius=3, color=color, thickness=-1)
            
    for i, XY in enumerate(XYs):
        color = colours[i % len(colours)]
        new_pt = smooth_curve(XY)
        for point in new_pt:
            x, y = int(margin + point[0] * multi), int(margin + point[1] * multi)
            cv2.circle(combined, (x, y), radius=3, color=color, thickness=-1)
            
    return canvas

def smooth_curve(XY, num_points=00):
    x = XY[:, 0]
    y = XY[:, 1]
    t = np.arange(len(x)) 
    t_new = np.linspace(0, len(x) - 1, num_points)
    
    # Interpolate using cubic splines
    f_x = interp1d(t, x, kind='cubic')
    f_y = interp1d(t, y, kind='cubic')
    
    x_new = f_x(t_new)
    y_new = f_y(t_new)
    
    return np.vstack((x_new, y_new)).T

def label(img):
    imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thrash = cv2.adaptiveThreshold(imgGry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            if 0.8 <= aspectRatio <= 1.2:
                cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif 6 <= len(approx) <= 8:
            cv2.putText(img, "Polygon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('shapes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

csv_path = './problems/frag1.csv' 
points = read_csv(csv_path)
start_end_points = []

for path in points:
    start_end_points.extend(get_start_end_points(path))

tolerance = 2.0

merged_curves, new_start_end_points = merge_curves(points, start_end_points, tolerance)

print("Number of merged curves:", len(merged_curves))

# plot(merged_curves)
combined  = np.ones((800, 800, 3), dtype=np.uint8) * 255

for curve in merged_curves:
    img = draw([curve])
    
    label(img)
    
cv2.imshow('All', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()