
import os
import argparse

import cv2
import numpy as np

from lossy_contour_algorithm import encode_one_mask

# nohup python -u generate_lossy_masks.py --start_index 0 --end_index 24 --regen > genrate_lossy_mask_info.out&
def generate_lossy_mask(lossless_path, lossy_path, T_init=5, thread=50, rate=0.15,
                        min_area=50):
    
    mask = cv2.imread(lossless_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask {lossless_path}')
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(os.path.dirname(lossy_path), exist_ok=True)
    tmp_path = lossy_path + '.tmp_region.png'
    merged = np.zeros_like(binary, dtype=np.uint8)
    total_bits = 0
    n_encoded = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        region_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, color=255, thickness=-1)
        cv2.imwrite(tmp_path, region_mask)
        bits, orig_bits, err, recon = encode_one_mask(
            tmp_path, T_init=T_init, thread=thread, rate=rate)
        merged = cv2.bitwise_or(merged, (recon > 0).astype(np.uint8) * 255)
        total_bits += int(bits)
        n_encoded += 1
        print(f'[lossy] {os.path.basename(lossless_path)} region {n_encoded}: '
              f'{bits} bits (orig {orig_bits}, err {err}px)')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cv2.imwrite(lossy_path, merged)
    with open(lossy_path.replace('.png', '_bits.txt'), 'w') as f:
        f.write(f'{total_bits}\n')
    print(f'[lossy] saved {lossy_path}  ({n_encoded} regions, {total_bits} contour bits)')
    return total_bits


def compute_iou(lossless_path, lossy_path):
    """Foreground IoU between the lossless mask and its lossy reconstruction."""
    lossless = cv2.imread(lossless_path, cv2.IMREAD_GRAYSCALE)
    lossy = cv2.imread(lossy_path, cv2.IMREAD_GRAYSCALE)
    if lossless is None or lossy is None:
        raise FileNotFoundError(f'Cannot read {lossless_path} or {lossy_path}')
    a = lossless > 127
    b = lossy > 127
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def main():
    parser = argparse.ArgumentParser(description='Generate lossy contour masks and report bits/IoU')
    parser.add_argument('--mask_dir', type=str,
                        default='./dataset/kodak_data_set/kodak_mask',
                        help='Merged binary masks (kodim{NN}.png).')
    parser.add_argument('--lossy_mask_dir', type=str,
                        default='./dataset/kodak_data_set/kodak_lossy_mask',
                        help='Output dir for lossy masks (+ _bits.txt sidecars).')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=24)
    parser.add_argument('--regen', action='store_true', default=False,
                        help='Re-encode even if a cached lossy mask exists.')
    parser.add_argument('--T_init', type=int, default=5)
    parser.add_argument('--thread', type=int, default=50)
    parser.add_argument('--rate', type=float, default=0.15)
    parser.add_argument('--min_area', type=int, default=50)
    args = parser.parse_args()

    names, bits_list, iou_list = [], [], []
    for it in range(args.start_index, args.end_index):
        idx_str = f'{it + 1:02d}'
        name = f'kodim{idx_str}'
        lossless_path = os.path.join(args.mask_dir, f'{name}.png')
        lossy_path = os.path.join(args.lossy_mask_dir, f'{name}.png')
        bits_path = lossy_path.replace('.png', '_bits.txt')

        if not os.path.exists(lossless_path):
            print(f'[skip] {lossless_path} not found')
            continue

        if args.regen or not os.path.exists(lossy_path) or not os.path.exists(bits_path):
            total_bits = generate_lossy_mask(
                lossless_path, lossy_path,
                T_init=args.T_init, thread=args.thread,
                rate=args.rate, min_area=args.min_area)
        else:
            with open(bits_path) as f:
                total_bits = int(f.read().strip())
            print(f'[lossy] reusing {lossy_path} ({total_bits} contour bits)')

        iou = compute_iou(lossless_path, lossy_path)
        names.append(name)
        bits_list.append(total_bits)
        iou_list.append(iou)
        print(f'{name}: bits = {total_bits}, IoU = {iou:.4f}')

    if not names:
        print('No masks processed.')
        return

    print('\n================ SUMMARY ================')
    print(f'{"image":<10} {"bits":>8} {"IoU":>8}')
    for name, bits, iou in zip(names, bits_list, iou_list):
        print(f'{name:<10} {bits:>8d} {iou:>8.4f}')
    print('-----------------------------------------')
    print(f'{"AVG":<10} {np.mean(bits_list):>8.1f} {np.mean(iou_list):>8.4f}')
    print(f'processed {len(names)} images')


if __name__ == '__main__':
    main()
