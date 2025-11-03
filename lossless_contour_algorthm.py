import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import numpy as np
from collections import deque
import math
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets, transforms

def mm(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (h, w)
    target_mask = mask == 255
    target_mask_flat = target_mask.flatten()
    target_mask_tensor = torch.from_numpy(target_mask_flat).bool()

    return target_mask_tensor, torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0)

def compute_vcc(vertices, mask):
    
    vcc_code = []
    vertex_count = {} 
    
  
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
               
                vertices_set = {(x, y), (x+1, y), (x, y+1), (x+1, y+1)}
                for v in vertices_set:
                    if v in vertex_count:
                        vertex_count[v] += 1
                    else:
                        vertex_count[v] = 1

    
    for v in vertices:
        v = tuple(v)  
        vcc_code.append(min(vertex_count.get(v, 0), 3))  

    return vcc_code

def compute_3ot(contour):
   
    directions = []
    num_points = len(contour)

   
    for i in range(num_points):
        next_idx = (i + 1) % num_points 
        dx = contour[next_idx][0] - contour[i][0]
        dy = contour[next_idx][1] - contour[i][1]

        if dx == 1 and dy == 0:  # →
            directions.append(0)
        elif dx == -1 and dy == 0:  # ←
            directions.append(2)
        elif dx == 0 and dy == -1:  # ↑
            directions.append(1)
        elif dx == 0 and dy == 1:  # ↓
            directions.append(1)
        else:  
            directions.append(2)

    
    three_ot_code = [0]  
    for i in range(1, len(directions)):
        if directions[i] == directions[i - 1]:
            three_ot_code.append(0)
        elif i > 1 and directions[i] == directions[i - 2]:
            three_ot_code.append(1)
        else:
            three_ot_code.append(2)

    return three_ot_code

def compute_nad(contour, initial_angle=0):
   
    num_points = len(contour)
    if num_points < 2:
        return [] 
    
    angles = []  
    nad_code = [] 

    
    for i in range(num_points):
        x1, y1 = contour[i][0]
        x2, y2 = contour[(i + 1) % num_points][0]  
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(-dy, dx))  
        angles.append(angle)

    
    prev_angle = initial_angle  

  
    for i in range(num_points):
        diff = angles[i] - prev_angle
        diff = ((diff + 180) % 360) - 180 

     
        if diff == 0:
            nad_code.append(0)
        elif diff == 45:
            nad_code.append(1)
        elif diff == -45:
            nad_code.append(2)
        elif diff == 90:
            nad_code.append(300)
        elif diff == -90:
            nad_code.append(310)
        elif diff == 135:
            nad_code.append(301)
        elif diff == -135:
            nad_code.append(311)
        else:  
            nad_code.append(312)

        prev_angle = angles[i]  
    split_data = [list(map(int, str(x))) if x > 9 else [x] for x in nad_code]

    
    flattened_data = [digit for sublist in split_data for digit in sublist]
    return flattened_data


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
    
    specified_start_point = (1, 9) 


    start_index = find_closest_point(contour, specified_start_point)


    contour_reordered = np.roll(contour, -start_index, axis=0) 
    x, y = contour_reordered[0][0]
   

    return contour_reordered
def get_outer_vertices(mask, contour, start_point=None):
    
    height, width = mask.shape
    vertex_count = {}  
    all_vertices = []  

 
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:  
                vertices = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                
                for v in vertices:
                    if v in vertex_count:
                        vertex_count[v] += 1
                    else:
                        vertex_count[v] = 1

   
    for point in contour:
        x, y = point[0]
        vertices = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

        for v in vertices:
            if vertex_count.get(v, 0) < 4: 
                all_vertices.append(v)

   
    all_vertices = list(dict.fromkeys(all_vertices))
    all_vertices_set = set(all_vertices) 

   
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # left → up → right → down

    
    if start_point==None:
        start_point=tuple(all_vertices[0])
    current_point = start_point

    ordered_vertices = [current_point] 
   
    visited_edges = set()

   
    while True:
        found_next = False
        
        for dx, dy in directions:
            next_point = (current_point[0] + dx, current_point[1] + dy)
            
            edge = (current_point, next_point)
            if next_point in all_vertices_set and  edge not in visited_edges:
                
                if dx == -1 and dy == 0:  
                    left_x, left_y = current_point[0] - 1, current_point[1]
                    right_x, right_y = current_point[0]-1, current_point[1]-1
                elif dx == 1 and dy == 0:  
                    left_x, left_y = current_point[0], current_point[1]-1
                    right_x, right_y = current_point[0], current_point[1]
                elif dx == 0 and dy == -1:  
                    left_x, left_y = current_point[0]-1, current_point[1]-1
                    right_x, right_y = current_point[0], current_point[1]-1
                elif dx == 0 and dy == 1:  
                    left_x, left_y = current_point[0], current_point[1]
                    right_x, right_y = current_point[0]-1, current_point[1]
            
               
                left_value = mask[left_y, left_x] if 0 <= left_x < width and 0 <= left_y < height else None
                right_value = mask[right_y, right_x] if 0 <= right_x < width and 0 <= right_y < height else None

               
                if left_value != right_value and left_value == 255:
                    ordered_vertices.append(next_point)
                    visited_edges.add(edge)                 
                    visited_edges.add((next_point, current_point))
                    current_point = next_point
                   
                    found_next = True
                    break

        if not found_next:
            break  

    return np.array(ordered_vertices)

def mtft_encode(stream, iterations=4, method='None'):
    
    
    results = []
    
    current_stream = stream[:]  
    results.append((current_stream, compute_entropy(current_stream)))
    for itera in range(iterations):
        unique_symbols = sorted(set(current_stream))  
        L = unique_symbols[:]  
        if itera==0 and method=='VCC':
            L=[2,1,3]
       
        encoded_stream = []
        
        for s in current_stream:
            index = L.index(s) 
            encoded_stream.append(index)
            L.pop(index)  
            L.insert(0, s)  
        
        results.append((encoded_stream, compute_entropy(encoded_stream)))
        current_stream = encoded_stream  
    
    return results

def compute_entropy(stream):
    
    counts = Counter(stream)
    total = len(stream)
    entropy = -sum((count / total) * np.log2(count / total) for count in counts.values())
    return entropy

def encode_arle(symbols,label=None):
    
    encoded_bits = ""
    i = 0
    while i < len(symbols):
        run_length = 1

        
        if symbols[i] == 0:
            while i + run_length < len(symbols) and symbols[i + run_length] == 0:
                run_length += 1
            if run_length >= 8:
                if run_length>37:
                    run_length=37
                encoded_bits += "1110" 
                encoded_bits += encode_vlc(run_length, mode=1)
                i += run_length
                continue  

        
        max_pattern_length = 16  
        arle2_triggered = False  
        for pattern_length in range(1, max_pattern_length + 1):
            window = symbols[i : i + pattern_length] 
            if len(window) < pattern_length:
                break  
            repetitions = 1
            while i + repetitions * pattern_length < len(symbols) and \
                  symbols[i + repetitions * pattern_length : i + (repetitions + 1) * pattern_length] == window:
                repetitions += 1
            
            r = sum(len(encode_symbol(s)) for s in window)  
            if r == 0:
                
                continue
            nomal_bits = r*repetitions
            a=11+len(encode_v_vcc(window))
            C = math.ceil(a / r)
            if repetitions > C and a < nomal_bits:
                remaining = repetitions
               
                while remaining > C:
                    reps = min(remaining, C + 30)
                    if reps < C + 1:
                        reps = C + 1

                   
                    encoded_bits += ("11110" if label == 'NAD' else "1111")
                    
                    encoded_bits += encode_vlc(len(window), mode=2)
                    
                    encoded_bits += encode_v_vcc(window)
                   
                    encoded_bits += encode_vlc(reps, mode=3, C=C)

                    i += reps * pattern_length
                    remaining -= reps

                
                if remaining > 0:
                    for _ in range(remaining):
                        for s in window:
                            encoded_bits += encode_symbol(s)
                    i += remaining * pattern_length

                arle2_triggered = True
                break
        if not arle2_triggered:
            encoded_bits += encode_symbol(symbols[i])
            i += 1 
    
    return encoded_bits

def encode_symbol(symbol):
    
    table = {
        0: "0",
        1: "10",
        2: "110",
        3: "11111",
        "ARLE1": "1110",
        "ARLE2": "1111",
    }
    return table.get(symbol, "") 

def encode_v_vcc(symbols):

    table = {
        0: "0",
        1: "10",
        2: "11",
        3: "01",
    }
    return "".join(table[s] for s in symbols)

def encode_vlc(value, mode, C=0):
   
    if mode == 1:  
        if 8 <= value <= 9:
            return "00" + ("0" if value == 8 else "1")  # 1-bit value
        elif 10 <= value <= 13:
            return "01" + "{:02b}".format(value - 10)  # 2-bit value
        elif 14 <= value <= 21:
            return "10" + "{:03b}".format(value - 14)  # 3-bit value
        elif 22 <= value <= 37:
            return "11" + "{:04b}".format(value - 22)  # 4-bit value
        else:
            raise ValueError(f"Unsupported run length: {value}")  
    elif mode == 2:  # ARLE2 for repeated patterns
        return "{:04b}".format(value - 1)  # Length |R| stored in 4 bits
    elif mode == 3:  # ARLE2 for repeated patterns
        if C+1 <= value <= C+2:
            return "00" + ("0" if value == C+1 else "1")  # 1-bit value
        elif C+3 <= value <= C+6:
            return "01" + "{:02b}".format(value - (C+3))  # 2-bit value
        elif C+7 <= value <= C+14:
            return "10" + "{:03b}".format(value - (C+7))  # 3-bit value
        elif C+15 <= value <= C+30:
            return "11" + "{:04b}".format(value - (C+15))  # 4-bit value
        else:
            # return "11" + "{:04b}".format(value - (C+15)) 
            raise ValueError(f"Unsupported run length: {value}{C}") 
    return ""



def get_border_bits(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    

    contour_reordered = contours[0]
    
    outer_vertices = get_outer_vertices(mask, contour_reordered)
   
    vcc_result = compute_vcc(outer_vertices, mask)
    
    nad_result = compute_nad(contour_reordered)
   
    three_ot_result = compute_3ot(outer_vertices)
   
    iterations = 4

  
    vcc_MTFT_results = mtft_encode(vcc_result, iterations,'VCC')
    nad_MTFT_results = mtft_encode(nad_result, iterations)
    three_ot_MTFT_results = mtft_encode(three_ot_result, iterations)

    vcc_best_encoding = None
    vcc_best_entropy = float('inf')
    best_type = ""
    nad_best_encoding = None
    nad_best_entropy = float('inf')
    three_ot_best_encoding = None
    three_ot_best_entropy = float('inf')
    for i, (encoded, entropy) in enumerate(vcc_MTFT_results):
        if entropy < vcc_best_entropy:
            vcc_best_encoding = encoded
            vcc_best_entropy = entropy
            best_type = f"VCC (Iteration {i})"
   
    for i, (encoded, entropy) in enumerate(nad_MTFT_results):
        if entropy < nad_best_entropy:
            nad_best_encoding = encoded
            nad_best_entropy = entropy
            best_type = f"NAD (Iteration {i})"
   
    for i, (encoded, entropy) in enumerate(three_ot_MTFT_results):
        if entropy < three_ot_best_entropy:
            three_ot_best_encoding = encoded
            three_ot_best_entropy = entropy
            best_type = f"3OT (Iteration {i})"
   


    compressed_bits_nad = encode_arle(nad_best_encoding,label='NAD')
    compressed_bits_vcc = encode_arle(vcc_best_encoding)
    compressed_bits_3ot = encode_arle(three_ot_best_encoding)
    
    compressed_bits = min(
        [compressed_bits_nad, compressed_bits_vcc, compressed_bits_3ot],
        key=len
    )
    number_compressed_bits = len(compressed_bits)+4
    

    return number_compressed_bits


