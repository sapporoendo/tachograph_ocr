import cv2
import numpy as np

img = cv2.imread('debug_result.png', cv2.IMREAD_GRAYSCALE)

if img is not None:
    height, width = img.shape
    center = (width // 2, height // 2)
    
    # マスク範囲（速度エリアをピンポイントで狙う）
    mask = np.zeros_like(img)
    inner_radius, outer_radius = int(min(center)*0.68), int(min(center)*0.82)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    img = cv2.bitwise_and(img, mask)

    max_radius = int(min(center) * 0.9)
    polar_img = cv2.warpPolar(img, (max_radius, 1440), center, max_radius, cv2.WARP_POLAR_LINEAR)
    
    # 速度エリアの平均的な強さを計算
    driving_intensity = np.sum(polar_img[:, int(max_radius*0.7):int(max_radius*0.82)], axis=1)
    
    # 【調整】しきい値を少し下げて、薄い線を拾いやすくする
    threshold = np.mean(driving_intensity) * 1.2 
    is_driving = driving_intensity > threshold

    # 【調整】抜けを埋める（10分以内の隙間は繋げる）
    # これにより、細切れの運転が一つの大きな塊になります
    refined_driving = is_driving.copy()
    gap_limit = 10 
    for i in range(1, 1440 - gap_limit):
        if is_driving[i-1]:
            for gap in range(1, gap_limit + 1):
                if is_driving[i + gap]:
                    for j in range(gap):
                        refined_driving[i + j] = True
                    break

    print("\n【改善後の判定結果】")
    in_driving, start_t = False, 0
    for i in range(1440):
        if refined_driving[i] and not in_driving:
            start_t, in_driving = i, True
        elif not refined_driving[i] and in_driving:
            duration = i - start_t
            if duration >= 10: # 10分以上の運転のみを表示
                h1, m1 = divmod(start_t, 60)
                h2, m2 = divmod(i, 60)
                print(f"運転：{h1:02d}:{m1:02d} 〜 {h2:02d}:{m2:02d} ({duration}分間)")
            in_driving = False
