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


def find_circle_contour_intersection(contour, center, radius, cursor, used_points2, dir):
   
    search_window = max(20, int(4 * radius))
    end_idx = min(cursor + search_window, len(contour))

    best_intersection = None
    min_distance_diff = float('inf')
    best_index = -1

   
    for i in range(cursor, end_idx):
        pt = tuple(contour[i][0])
        distance = np.linalg.norm(np.array(pt) - np.array(center))
        distance_diff = abs(distance - radius)
        if distance_diff < min_distance_diff:
            min_distance_diff = distance_diff
            best_intersection = pt
            best_index = i

    
    if best_intersection is None or min_distance_diff > radius / 3.0:
        return 'Finish', cursor, used_points2

  
    return best_intersection, best_index + 1, used_points2

def _build_directions(T, phi, center_angle, count):
   
    if phi >= 2 * np.pi - 1e-9:
        step = 2 * np.pi / count
        return [(round(T * np.cos(center_angle + i * step), 3),
                 round(T * np.sin(center_angle + i * step), 3))
                for i in range(count)]
    step = phi / (count - 1)
    base = center_angle - phi / 2
    return [(round(T * np.cos(base + i * step), 3),
             round(T * np.sin(base + i * step), 3))
            for i in range(count)]


def _circular_mean(angles):
   
    s = float(np.mean([np.sin(a) for a in angles]))
    c = float(np.mean([np.cos(a) for a in angles]))
    m = float(np.arctan2(s, c))
    return m + 2 * np.pi if m < 0 else m


def _angle_diff(a, b):
   
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _near(p, q, tol):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < tol * tol


def chain_code(contour, connectivity=8, T=10, M=8, N=16, phi_t=2*np.pi,
               global_reset=False, Adaptive_chain=True,
               phi_min=None, phi_max=None):
   
    if phi_min is None:
        phi_min = phi_t / 8.0
    if phi_max is None:
        phi_max = 2 * np.pi
    original_phi_t = phi_t

    start_point = tuple(contour[0][0])
    current_point = start_point
    cursor = 0                  
    used_points2 = set()        
    chain = []
    location = [current_point]
    full_dir = []
    past_angles = []
    selected_angle = 0.0
    first_step = True
    just_doubled = False        
    step_count = 0
   
    max_steps = max(len(contour) * 2 + 10, 4 * len(contour) // max(T, 1) + 50)

    while step_count < max_steps:
        step_count += 1

        intersection, cursor, used_points2 = find_circle_contour_intersection(
            contour, current_point, T, cursor, used_points2, 0
        )
        if intersection == 'Finish':
            break

       
        if first_step:
            init_dirs = _build_directions(T, 2 * np.pi, 0.0, N)
            init_cands = [(current_point[0] + d[0], current_point[1] + d[1]) for d in init_dirs]
            best_dir = find_best_direction(current_point, intersection, init_dirs)
            chain.append(best_dir)
            full_dir.append(best_dir)
            current_point = init_cands[best_dir]
            d = init_dirs[best_dir]
            selected_angle = float(np.arctan2(d[1], d[0]))
            if selected_angle < 0:
                selected_angle += 2 * np.pi
            past_angles.append(selected_angle)
            location.append(current_point)
            first_step = False
            continue

       
        dirs = _build_directions(T, phi_t, selected_angle, M)
        cands = [(current_point[0] + d[0], current_point[1] + d[1]) for d in dirs]
       
        best_dir = find_best_direction(current_point, intersection, dirs)
        distance_err = float(np.linalg.norm(np.array(intersection) - np.array(cands[best_dir])))

       
        reset_thresh = 2 * T * np.sin(original_phi_t / (4 * (M - 1)))
        if (Adaptive_chain or global_reset) and distance_err > reset_thresh:
            chain.append('x')
            reset_dirs = _build_directions(T, 2 * np.pi, 0.0, N)
            reset_cands = [(current_point[0] + d[0], current_point[1] + d[1]) for d in reset_dirs]
            best_dir = find_best_direction(current_point, intersection, reset_dirs)
            chain.append(best_dir)
            full_dir.append(best_dir)
            current_point = reset_cands[best_dir]
            d = reset_dirs[best_dir]
            selected_angle = float(np.arctan2(d[1], d[0]))
            if selected_angle < 0:
                selected_angle += 2 * np.pi
          
            phi_t = original_phi_t
            past_angles = [selected_angle]
            just_doubled = False
            location.append(current_point)
           
            if cursor > len(contour) // 2 and _near(current_point, start_point, T):
                break
            continue

        
        chain.append(best_dir)
        full_dir.append(best_dir)
        current_point = cands[best_dir]
        d = dirs[best_dir]
        selected_angle = float(np.arctan2(d[1], d[0]))
        if selected_angle < 0:
            selected_angle += 2 * np.pi
        past_angles.append(selected_angle)
        location.append(current_point)

       
        if step_count > 1 and _near(current_point, start_point, T):
            break

      
        if Adaptive_chain:
            if just_doubled:
               
                just_doubled = False
            elif len(past_angles) >= 2:
               
                avg = _circular_mean(past_angles)
                diff = abs(_angle_diff(avg, selected_angle))
                spacing = phi_t / (M - 1)
                if diff < 0.5 * spacing:
                   
                    phi_t = max(phi_t * 0.5, phi_min)
                elif diff > spacing:
                    new_phi = min(phi_t * 2.0, phi_max)
                    if new_phi > phi_t:
                        phi_t = new_phi
                        past_angles = []     
                        just_doubled = True  

    return chain, location
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
        # dx, dy = location[i]
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
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # Convert to binary mask

   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]  # Assuming single contour
    
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

def _parse_info_txt(info_path):
   
    entries = []
    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            entries.append((parts[0], parts[-1]))
    return entries


def encode_one_mask(mask_path, T_init=5, thread=10, rate=0.3,
                    phi_t=np.pi / 8, M=3):
    
    T = T_init
    iteration = 0
    while True:
        bits, original_cost, error_8, filled_reconstructed = \
            get_lossy_border_bits(mask_path, 0, T,
                                  phi_t=phi_t, M=M, iteration=iteration)
        if error_8 <= thread and bits <= original_cost * rate:
            return bits, original_cost, error_8, filled_reconstructed
        if bits > original_cost * rate:
            T += 1
        if error_8 > thread:
            iteration += 1


def encode_image_K(it, K,
                   mask_root="dataset/kodak_data_set/kodak_mask",
                   out_root="dataset/kodak_data_set/kodak_lossy_mask",
                   T_init=5, thread=10, rate=0.3, save=True):
   
    idx_str = f"{it:02d}"
    in_dir = os.path.join(mask_root, f"kodim{idx_str}", f"K{K}")
    info_path = os.path.join(in_dir, "info.txt")
    if not os.path.exists(info_path):
        raise FileNotFoundError(info_path)
    entries = _parse_info_txt(info_path)
    encoded = [f for f, r in entries if r == "encoded"]

    out_dir = os.path.join(out_root, f"kodim{idx_str}", f"K{K}")
    if save:
        os.makedirs(out_dir, exist_ok=True)

    total_bits = 0
    total_orig = 0
    per_mask = []
    for fname in encoded:
        mask_path = os.path.join(in_dir, fname)
        if not os.path.exists(mask_path):
            print(f"[warn] missing {mask_path}, skipping")
            continue
        print(f"--- kodim{idx_str} K{K} {fname} ---")
        bits, orig, err, recon = encode_one_mask(
            mask_path, T_init=T_init, thread=thread, rate=rate
        )
        total_bits += bits
        total_orig += orig
        per_mask.append((fname, bits, orig, err))
        if save:
            cv2.imwrite(os.path.join(out_dir, fname),
                        (recon > 0).astype(np.uint8) * 255)

    if save:
       
        for fname, role in entries:
            if role == "encoded":
                continue
            src = os.path.join(in_dir, fname)
            if os.path.exists(src):
                cv2.imwrite(os.path.join(out_dir, fname),
                            cv2.imread(src, cv2.IMREAD_GRAYSCALE))
        with open(os.path.join(out_dir, "info.txt"), "w") as f:
            f.write("# region_{k}.png  bits  original_bits  error  role\n")
            for fname, bits, orig, err in per_mask:
                f.write(f"{fname}  {bits}  {orig}  {err}  encoded\n")
            for fname, role in entries:
                if role != "encoded":
                    f.write(f"{fname}  -  -  -  {role}\n")
            f.write(f"# total_encoded_bits  {total_bits}\n")
            f.write(f"# total_original_bits {total_orig}\n")

    print(f"[ok] kodim{idx_str} K{K}: total_bits={total_bits} "
          f"(orig={total_orig})")
    return total_bits


def run_kodak_lossy(Ks=(2, 3, 4, 5), indices=range(1, 25),
                    mask_root="dataset/kodak_data_set/kodak_mask",
                    out_root="dataset/kodak_data_set/kodak_lossy_mask",
                    T_init=5, thread=10, rate=0.3, save=True):
   
    indices = list(indices)
    results = {}
    for K in Ks:
        per_image = {}
        for it in indices:
            per_image[it] = encode_image_K(
                it, K, mask_root=mask_root, out_root=out_root,
                T_init=T_init, thread=thread, rate=rate, save=save,
            )
        avg = sum(per_image.values()) / max(len(per_image), 1)
        results[K] = {**per_image, "_avg": avg}
        print(f"\n[K={K}] per-image bits: "
              + ", ".join(f"kodim{it:02d}={b}" for it, b in per_image.items()))
        print(f"[K={K}] average bits per image: {avg:.1f}\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=str,
                        default="dataset/kodak_data_set/kodak_mask")
    parser.add_argument("--out_root", type=str,
                        default="dataset/kodak_data_set/kodak_lossy_mask")
    parser.add_argument("--Ks", type=str, default="2,3,4,5")
    parser.add_argument("--indices", type=str, default="",
                        help="comma-separated kodim indices (1..24); empty=all")
    parser.add_argument("--T_init", type=int, default=5)
    parser.add_argument("--thread", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--no_save", action="store_true",
                        help="skip writing reconstructed masks")
    args = parser.parse_args()

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    indices = ([int(x) for x in args.indices.split(",") if x.strip()]
               if args.indices.strip() else list(range(1, 25)))

    run_kodak_lossy(Ks=Ks, indices=indices,
                    mask_root=args.mask_root, out_root=args.out_root,
                    T_init=args.T_init, thread=args.thread, rate=args.rate,
                    save=not args.no_save)

# python lossy_contour_algorithm.py --Ks 2,3,4,5 --indices 1,4,10