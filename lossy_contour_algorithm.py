import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2
import numpy as np
from collections import deque
import math
import torch
import os
from collections import Counter
def find_best_direction(center, intersection, directions):
   
    cx, cy = center
    ix, iy = intersection
    

    v = np.array([ix - cx, iy-cy])
    
   
    min_angle = float("inf")
    best_idx = -1
    
    for i, (dx, dy) in enumerate(directions):
        d = np.array([dx, dy])
        
        
        dot_product = np.dot(v, d)
        
        
        norm_v = np.linalg.norm(v)
        norm_d = np.linalg.norm(d)
        
        if norm_v == 0 or norm_d == 0:  
            continue
        
       
        cos_theta = np.clip(dot_product / (norm_v * norm_d), -1.0, 1.0)
        angle = np.arccos(cos_theta) 
        
      
        if angle < min_angle:
            min_angle = angle
            best_idx = i
    
    return best_idx

def find_circle_contour_intersection(contour, center, radius, used_points, used_points2, dir):
   
    cx, cy = center
    best_intersection = None
    min_distance_diff = float('inf')
    predicted_points = used_points.copy()
    calculated_num = 0
   
    filtered_contour = np.array([pt for pt in contour if tuple(pt[0]) not in used_points])

   
    for i in range(1,len(filtered_contour)):
        
        pt1 = tuple(filtered_contour[i-1][0])
       
        distance = np.linalg.norm(np.array(pt1) - np.array(center))  
        distance_diff = abs(distance - radius)  
        calculated_num += 1
       
        if  distance_diff < min_distance_diff and distance_diff < 1/3*radius:
        
            min_distance_diff = distance_diff
            best_intersection = pt1
            index = i
        if calculated_num >20:
            break
    if best_intersection is not None:
        for i in range(1,index):
            pt1 = tuple(filtered_contour[i-1][0])
            if pt1 not in predicted_points:
                    predicted_points.add(pt1)



  
    if best_intersection is not None:
        used_points = predicted_points.copy()
        
        return best_intersection, used_points, used_points2

   
    best_intersection = 'Finish'
    return best_intersection, used_points, used_points2

def chain_code(contour, connectivity=8, T=10, M=8,N=16, phi_t=2*np.pi, global_reset=False, Adaptive_chain=True):
    
    if M%2==0:
        theta_t = phi_t / M
    else:
        theta_t = phi_t / (M-1)
    directions = [(round(T *np.cos(i * theta_t),3), round(T *np.sin(i * theta_t),3)) for i in range(M)]
    original_phi_t = phi_t
   
    current_point = tuple(contour[0][0])
    used_points = set()  
    chain = []
    location = []
    location.append(current_point)
    best_dir = None
    full_dir = []
    stuck_counter = 0  
    max_stuck = 15 
   
    used_points2 = set()
    dir=0
    last_point = None
    first_step = True
    reset = False
    index = 0
    past_angle_degrees = []
    while True:
        index += 1
       
        intersection, used_points, used_points2 = find_circle_contour_intersection(contour, current_point, T, used_points, used_points2, dir)
        if intersection == 'Finish':
            break
        last_point = current_point
        if first_step:
            theta_f_s = 2*np.pi/16
            directions1 = [(round(T *np.cos(i * theta_f_s),3), round(T *np.sin(i * theta_f_s),3)) for i in range(16)]
            candidates1 = [(current_point[0] + d[0], current_point[1] + d[1]) for d in directions1]
            best_dir = find_best_direction(current_point, intersection, directions1)
            full_dir.append(best_dir)
            chain.append(best_dir)
            current_point = candidates1[best_dir]  
            first_direction_vector = directions1[best_dir]
            selected_angle = np.arctan2(first_direction_vector[1], first_direction_vector[0]) 
            if selected_angle < 0:
                selected_angle += 2 * np.pi
            selected_angle_degrees = np.degrees(selected_angle)
            first_step = False
            past_angle_degrees.append(selected_angle)
                            
        else:
            if Adaptive_chain:
                    min_angle = np.arctan2(directions[0][1], directions[0][0])
                    max_angle = np.arctan2(directions[-1][1], directions[-1][0])
                    if min_angle < 0:
                        min_angle += 2 * np.pi
                    if max_angle < 0:
                        max_angle += 2 * np.pi
                    mid_angle = (max_angle+min_angle) / 2
                    selected_angle_degrees = np.degrees(mid_angle)
                    angle_offset = selected_angle - mid_angle
                    
                    rotated_directions = [(round(d[0] * np.cos(angle_offset) - d[1] * np.sin(angle_offset), 3), 
                                round(d[0] * np.sin(angle_offset) + d[1] * np.cos(angle_offset), 3)) 
                                for d in directions]
                    
                    candidates1 = [(current_point[0] + d[0], current_point[1] + d[1]) for d in rotated_directions]
                   
                    best_candidate = min(candidates1, key=lambda p: np.linalg.norm(np.array(p) - np.array(intersection)))
                    best_dir = candidates1.index(best_candidate)  
                    distance_err = np.linalg.norm(np.array(intersection) - np.array(candidates1[best_dir]))
                    if abs(distance_err) > 2*T*np.sin(phi_t/(4*(M-1))):
                        theta_global = 2*np.pi/N
                        chain.append('x')
                        directions = [(round(T *np.cos(i * theta_global),3), round(T *np.sin(i * theta_global),3)) for i in range(N)]
                        reset = True
                        
            if reset:
                phi_t = original_phi_t
                candidates1 = [(current_point[0] + d[0], current_point[1] + d[1]) for d in directions]
              
                best_dir = find_best_direction(current_point, intersection, directions)
                full_dir.append(best_dir)
                
                chain.append(best_dir)
                current_point = candidates1[best_dir] 
                first_direction_vector = directions[best_dir]
                selected_angle = np.arctan2(first_direction_vector[1], first_direction_vector[0])  
                if selected_angle < 0:
                    selected_angle += 2 * np.pi
                reset = False
                directions = [(round(T *np.cos(i * theta_t),3), round(T *np.sin(i * theta_t),3)) for i in range(M)]
                past_angle_degrees.append(selected_angle)
                used_contour = np.squeeze(contour)
                distances = np.linalg.norm(used_contour - np.array(current_point), axis=1)
                
                
            else:
                
                min_angle = np.arctan2(directions[0][1], directions[0][0])
                max_angle = np.arctan2(directions[-1][1], directions[-1][0])
                if min_angle < 0:
                    min_angle += 2 * np.pi
                if max_angle < 0:
                    max_angle += 2 * np.pi
                mid_angle = (max_angle+min_angle) / 2
                selected_angle_degrees = np.degrees(mid_angle)
                angle_offset = selected_angle - mid_angle
                
                rotated_directions = [(round(d[0] * np.cos(angle_offset) - d[1] * np.sin(angle_offset), 3), 
                            round(d[0] * np.sin(angle_offset) + d[1] * np.cos(angle_offset), 3)) 
                            for d in directions]
                directions = rotated_directions
                candidates1 = [(current_point[0] + d[0], current_point[1] + d[1]) for d in directions]
                
                best_dir = find_best_direction(current_point, intersection, directions)
                full_dir.append(best_dir)
                
                chain.append(best_dir)
                current_point = candidates1[best_dir]  
                first_direction_vector = directions[best_dir]
                selected_angle = np.arctan2(first_direction_vector[1], first_direction_vector[0])  
                if selected_angle < 0:
                    selected_angle += 2 * np.pi
                past_angle_degrees.append(selected_angle)
                dir = 0
                if Adaptive_chain:

                    avg_past_angle = sum(past_angle_degrees)/(index)
                    if abs(avg_past_angle-selected_angle) < 1/2*(phi_t/(M-1)):
                        
                        theta_a = (1/2*phi_t)/(M-1)
                      
                        directions = [(round(T *np.cos(i * theta_a),3), round(T *np.sin(i * theta_a),3)) for i in range(M)]
                        
                    elif abs(avg_past_angle-selected_angle) > (phi_t/(M-1)):
                       
                        past_angle_degrees = []
                        index = 0
                        theta_a = (2*phi_t)/(M-1)
                        directions = [(round(T *np.cos(i * theta_a),3), round(T *np.sin(i * theta_a),3)) for i in range(M)]
                    
        
        location.append(current_point)

      
           
    return chain,location
def compute_error(original_mask, reconstructed_mask):
   
    original_mask = (original_mask > 0).astype(np.uint8)
    reconstructed_mask = (reconstructed_mask > 0).astype(np.uint8)

   
    contours, _ = cv2.findContours(reconstructed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_reconstructed = np.zeros_like(reconstructed_mask)
    cv2.drawContours(filled_reconstructed, contours, -1, 1, thickness=-1) 

   
    missing_pixels = np.sum((original_mask == 1) & (filled_reconstructed == 0))
    missing_pixels2 = np.sum((original_mask == 0) & (filled_reconstructed == 1))
    return missing_pixels,missing_pixels2,filled_reconstructed


def reconstruct_mask(start_point, chain_code,location, connectivity=8, T=10, mask_shape=(512, 512)):
    reconstructed = np.zeros(mask_shape, dtype=np.uint8)
    x, y = start_point
    reconstructed[y, x] = 255 
    
   
    location.append(start_point)
    for i in range(len(location)-1):
        
        new_x, new_y = location[i+1]
        x_vals = np.linspace(x, new_x, 100)
        y_vals = np.linspace(y, new_y, 100)

       
        for x, y in zip(x_vals, y_vals):
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < mask_shape[0] and 0 <= ix < mask_shape[1]:
                reconstructed[iy, ix] = 255
       
        x, y = new_x, new_y
    
    return reconstructed
def generate_sine_wave_mask(amplitude=200, period=500, width=500, height=500, step=0.05):
    
    x = np.arange(0, period, step)
    y = amplitude * np.sin(2 * np.pi * x / period+ np.pi) + height // 2 

    mask = np.zeros((height, width), dtype=np.uint8)


    for i in range(len(x)):
        xi, yi = int(x[i]), int(y[i])
        if 0 <= xi < width and 0 <= yi < height:
            mask[yi, xi] = 255  

    return mask
def draw_grid(image, grid_size=10):
   
    height, width = image.shape
    grid = np.zeros((height, width), dtype=np.uint8)
   
    for x in range(0, width, grid_size):
        cv2.line(image, (x, 0), (x, height), (50, 50, 50), 1)

   
    for y in range(0, height, grid_size):
        cv2.line(image, (0, y), (width, y), (50, 50, 50), 1)

    return image

def find_closest_point(contour, target_point):
    min_dist = float("inf")
    start_index = 0
    for i, point in enumerate(contour):
        x, y = point[0]
        dist = (x - target_point[0])**2 + (y - target_point[1])**2  
        if dist < min_dist:
            min_dist = dist
            start_index = i
    return start_index
def get_point(contour):
 
    specified_start_point = (0, 250) 


   
    start_index = find_closest_point(contour, specified_start_point)

   
    contour_reordered = np.roll(contour, -start_index, axis=0) 
    x, y = contour_reordered[0][0]
  
    return contour_reordered

def generate_lossy_mask(lossy_mask, it):
   
    output_dir = 'lossy_mask'
    os.makedirs(output_dir, exist_ok=True)

    it += 1
  
    filename = f'kodim0{it}.png' if it < 10 else f'kodim{it}.png'
    filepath = os.path.join(output_dir, filename)

   
    binary_mask = (lossy_mask > 0).astype('uint8') * 255

    cv2.imwrite(filepath, binary_mask)

def get_lossy_border_bits(mask_path,it,T,phi_t=np.pi/8,M=3,iteration=1):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0] 
  
    if iteration>0:
        kernel = np.ones((3,3), np.uint8)
        
        mask_dilated = cv2.dilate(mask, kernel, iterations=iteration)
        
        contour_dilated, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contour_dilated[0]
        

    start_point = tuple(contour[0][0]) 

    chain_8,location = chain_code(contour, connectivity=8, T=T, M=M,N=16, phi_t=phi_t,global_reset=True, Adaptive_chain=True)
    print("chain_8 bits:",len(chain_8))
    reconstructed_8 = reconstruct_mask(start_point, chain_8,location, connectivity=8, T=T, mask_shape=mask.shape)
    error_8,extra_error8,filled_reconstructed = compute_error(mask, reconstructed_8)
    extra_bits = sum(1 for x in chain_8 if x == 'x')
    total_bits_cost = len(chain_8)*2+extra_bits
    print(f"contour length: {len(contour)} pixels, Normal encoding: {len(contour)*3}bits")
    print(f"8-connected encoding: {total_bits_cost}bits, Error: {error_8} pixels,extra including pixel Error: {extra_error8} pixels")
    
    overlay_8 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    overlay_8[mask == 255] = (255, 255, 255)

    overlay_8[reconstructed_8 == 255] = (255, 255, 0) 

    cv2.imwrite("overlay_8_direction.png", overlay_8)
    return total_bits_cost,len(contour)*3,error_8,filled_reconstructed

def get_border_bits(mask_path,it,T=5,thread=10,rate=0.3):
    sum_bits_num = 0
    sum_bits_rate= 0
    num_images = 0
    iteration=0
    finish = True
    save_mask = True
    while finish:
        total_bits_cost,original_cost,error_8,filled_reconstructed=get_lossy_border_bits(mask_path,it,T,phi_t=np.pi/8,M=3,iteration=iteration)
       
        if total_bits_cost>original_cost*rate:
            T+=1
        if error_8 > thread:
            iteration +=1
        if error_8 <= thread and total_bits_cost<=original_cost*rate:
            
            finish=False
    return total_bits_cost
