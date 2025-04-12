"""
画質評価の指標ライブラリ
===========================

本モジュールは、画像の画質評価に関連する各種指標を計算するための関数群を提供します。
以下の評価指標が含まれています：

- PSNR, SSIM 
- UIQM, UCIQE 
- L*の標準偏差
- CR (Clipping Rate）
- HD 

各関数は、基本的に0–255スケールの画像（uint8 あるいは float）を入力として想定しています。
"""

import numpy as np
import math
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import color, filters
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

###########################
# ここから参照型定量評価計算部分（PSNR, SSIM)
###########################
def rmetrics(ref, target):
    """
    参照画像 (ref) と対象画像 (target) の PSNR と SSIM を計算する関数。

    画像は 0–255 スケール (uint8) を前提とし、画像が多チャンネルの場合（例: RGB）の場合には
    channel_axis=2 を使用します。なお、SSIMのウィンドウサイズは画像サイズに合わせて自動調整されます。

    Parameters:
        ref (ndarray): 参照画像
        target (ndarray): 対象画像

    Returns:
        tuple: (PSNR値, SSIM値)
    """
    psnr_val = compare_psnr(ref, target, data_range=255)

    # 画像の高さと幅（チャネルは除く）を取得
    h_ref, w_ref = ref.shape[0], ref.shape[1]
    h_tgt, w_tgt = target.shape[0], target.shape[1]
    min_dim = min(h_ref, w_ref, h_tgt, w_tgt)

    # デフォルトのウィンドウサイズは 7 だが、画像が小さい場合は調整
    win_size = 7
    if min_dim < 7:
        # 画像の最小次元に合わせた奇数に設定（最小次元が偶数の場合は 1 引く）
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 1:
            win_size = 1

    # 画像が多チャンネル（RGBなど）の場合は channel_axis=2 を指定
    if ref.ndim == 3 and ref.shape[2] in [3, 4]:
        ssim_val = compare_ssim(ref, target, channel_axis=2, data_range=255, win_size=win_size)
    else:
        ssim_val = compare_ssim(ref, target, data_range=255, win_size=win_size)
    return psnr_val, ssim_val


###########################
# ここから非参照型定量評価計算部分（UIQM, UCIQE, L*の標準偏差）
###########################
def nmetrics(a):
    """
    画像 a の UIQM, UCIQE, および L* の標準偏差を計算する関数。

    UCIQEは内部で 0–1 スケールで計算されるため、画像 a の正規化が必要ですが、
    skimage の rgb2lab, rgb2gray は入力スケールをそのまま扱うため注意が必要です。
    UIQMは 0–255 スケールのままで計算されます。

    Parameters:
        a (ndarray): 入力画像 (uint8 または float)。通常は 0–255 のスケールを前提。

    Returns:
        tuple: (UIQM, UCIQE, L* の標準偏差)
    """
    # rgb, lab, gray の計算
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)

    # UCIQE の定数
    c1, c2, c3 = 0.4680, 0.2745, 0.2576

    # L*, a*, b* (lab 空間)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # 1) 彩度
    chroma = np.sqrt(A ** 2 + B ** 2)
    uc = np.mean(chroma)
    sc = np.sqrt(np.mean((chroma - uc) ** 2))

    # 2) コントラスト（上位1%と下位1%の輝度の差）
    top = int(round(0.01 * L.size))
    L_sorted = np.sort(L, axis=None)
    L_sorted_inv = L_sorted[::-1]
    conl = np.mean(L_sorted_inv[:top]) - np.mean(L_sorted[:top])

    # 3) 彩度/輝度比
    satur_list = []
    chroma_flat = chroma.flatten()
    L_flat = L.flatten()
    for cval, lval in zip(chroma_flat, L_flat):
        if cval == 0 or lval == 0:
            satur_list.append(0)
        else:
            satur_list.append(cval / lval)
    us = np.mean(satur_list)

    # 最終 UCIQE の計算
    uciqe = c1 * sc + c2 * conl + c3 * us

    # ---------------------
    # UIQM の計算
    # ---------------------
    p1, p2, p3 = 0.0282, 0.2953, 3.5753

    # 1) UICM の計算
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1, al2 = 0.1, 0.1
    T1 = int(al1 * len(rgl))
    T2 = int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2) UISM の計算
    Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    Rsobel = np.round(Rsobel).astype(np.uint8)
    Gsobel = np.round(Gsobel).astype(np.uint8)
    Bsobel = np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3) UIConM の計算
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm

    # ここから追加部分：L*の標準偏差の計算
    std_L = np.std(L)

    return uiqm, uciqe, std_L


###########################
# 補助関数（UIQM, UCIQE の計算に利用）
###########################
def eme(ch, blocksize=8):
    """
    入力チャネル ch に対して、ブロック単位で Enhancement Measure Estimation (EME) を計算する。

    Parameters:
        ch (ndarray): 評価対象の1チャネル画像 (2次元配列)
        blocksize (int, optional): ブロックサイズ。デフォルトは8。

    Returns:
        float: 計算された EME 値
    """
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme_val = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        xrb = (i + 1) * blocksize if i < num_x - 1 else ch.shape[0]

        for j in range(num_y):
            ylb = j * blocksize
            yrb = (j + 1) * blocksize if j < num_y - 1 else ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            if blockmin == 0:
                blockmin += 1
            if blockmax == 0:
                blockmax += 1
            eme_val += w * math.log(blockmax / blockmin)
    return eme_val


def plipsum(i, j, gamma=1026):
    """
    PLIP (Perceived Luminance Image Processing) の加算操作を実行する。

    Parameters:
        i (float or ndarray): 入力値
        j (float or ndarray): 入力値
        gamma (float, optional): 定数。デフォルトは1026。

    Returns:
        float or ndarray: i と j を PLIP 加算した結果
    """
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    """
    PLIP における減算操作を実行する。

    Parameters:
        i (float or ndarray): 被減算値
        j (float or ndarray): 減算する値
        k (float, optional): 定数。デフォルトは1026。

    Returns:
        float or ndarray: PLIP 減算の結果
    """
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    """
    PLIP における乗算操作を実行する。

    Parameters:
        c (float or ndarray): 乗数
        j (float or ndarray): 入力値
        gamma (float, optional): 定数。デフォルトは1026。

    Returns:
        float or ndarray: PLIP 乗算の結果
    """
    return gamma - gamma * (1 - j / gamma) ** c


def logamee(ch, blocksize=8):
    """
    入力のグレースケールチャネルに対して、対数重み付けされた EME (log-EME) を計算する。

    この関数は UIConM の算出で利用されます。

    Parameters:
        ch (ndarray): 入力画像のグレースケールチャネル (2次元配列)
        blocksize (int, optional): ブロックサイズ。デフォルトは8。

    Returns:
        float: 計算された対数重み付け EME 値
    """
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        xrb = (i + 1) * blocksize if i < num_x - 1 else ch.shape[0]

        for j in range(num_y):
            ylb = j * blocksize
            yrb = (j + 1) * blocksize if j < num_y - 1 else ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))
            top = plipsub(blockmax, blockmin)
            epsilon = 1e-10  # ゼロ除算防止
            bottom = plipsum(blockmax, blockmin) + epsilon

            m = top / bottom
            if m != 0.:
                s += m * np.log(m)
    return plipmult(w, s)



###########################
# 箱ひげ図作成部分
###########################
def plot_boxplots(results, folder_list, selected_metrics, x_labels=None, save_dir=None):
    """
    選択された評価指標のみの箱ひげ図を作成し、指定されたディレクトリに保存する。

    Parameters:
        results (dict): 指標の計算結果が格納された辞書
        folder_list (list): フォルダパスのリスト
        selected_metrics (list): プロットする評価指標のリスト
        x_labels (list or str, optional): x軸に表示するラベル。指定がなければフォルダ名が用いられる。
        save_dir (str, optional): プロット画像の保存先ディレクトリ。指定されない場合は保存されない。

    Returns:
        None
    """
    plt.rcParams['font.family'] = 'MS Gothic'
    
    if x_labels is None or isinstance(x_labels, str):
        x_labels = [os.path.basename(folder) for folder in folder_list]
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    data = {metric: [] for metric in selected_metrics}
    
    for folder in folder_list:
        for metric in selected_metrics:
            if folder in results and metric in results[folder]:
                data[metric].append(results[folder][metric])
            else:
                print(f"Warning: {folder} の {metric} のデータが存在しません。スキップします。")
    
    for metric in selected_metrics:
        if len(data[metric]) == 0:
            print(f"Warning: {metric} のデータが不足しているためプロットできません。")
            continue
        
        plt.figure(figsize=(6, 5))
        plt.boxplot(data[metric], showmeans=True)
        plt.xticks(range(1, len(x_labels) + 1), x_labels, rotation=0)
        plt.title(f"{metric} Boxplot")
        plt.ylabel(metric)
        plt.grid(True)
        
        save_path = os.path.join(save_dir, f"metrics_boxplot_{metric}.png") if save_dir else f"metrics_boxplot_{metric}.png"
        plt.savefig(save_path)
        plt.show()


###########################
# CR (Clipping Rate) の計算
###########################
def compute_cr(ref, target):
    """
    CR (Clipping Rate) を計算する関数。
    入力画像において、処理前後で白または黒に飽和したピクセルの数の差異の割合を計算します。

    Parameters:
        ref (ndarray): 参照画像
        target (ndarray): 対象画像

    Returns:
        float: CR
    """
    ref = np.array(ref, dtype=float)
    target = np.array(target, dtype=float)
    
    if ref.ndim == 3 and ref.shape[2] >= 3:
        sum_ref = np.sum(ref[..., :3], axis=2)
        sum_target = np.sum(target[..., :3], axis=2)
        white_val = 255 * 3  # 白の閾値
    else:
        sum_ref = ref
        sum_target = target
        white_val = 255

    ind_w_ref = (sum_ref == white_val)
    ind_w_target = (sum_target == white_val)
    ind_w = (ind_w_ref != ind_w_target)
    
    ind_k_ref = (sum_ref == 0)
    ind_k_target = (sum_target == 0)
    ind_k = (ind_k_ref != ind_k_target)
    
    m, n = ref.shape[0], ref.shape[1]
    CR = (np.sum(ind_w) + np.sum(ind_k)) / (m * n)
    return CR


###########################################
# HD の計算 (Raines の定義および Lab 空間での計算)
###########################################
def f_rgb2hsi_raines(r, g=None, b=None):
    """
    RGB画像またはRGB値を、Rainesの定義に基づくHSIに変換する関数。

    入力は、各チャンネルが [0, 255] または [0, 1] の配列、もしくは Nx3 の配列として与えられます。

    Parameters:
        r (ndarray): 赤成分またはRGB画像全体
        g (ndarray, optional): 緑成分
        b (ndarray, optional): 青成分

    Returns:
        tuple: (h, s, i)
            h: Hue (0～2π、計算不可能な値はNaN)
            s: Saturation (0～2/3)
            i: Intensity (0～1)
    """
    if g is None and b is None:
        if r.dtype == np.uint8:
            r = r.astype(np.float64) / 255.0
        elif r.dtype == np.uint16:
            r = r.astype(np.float64) / 65535.0
        elif np.max(r) > 1.0:
            r = r.astype(np.float64) / 255.0

        if r.ndim == 3 and r.shape[2] == 3:
            g = r[:, :, 1]
            b = r[:, :, 2]
            r = r[:, :, 0]
            original_shape = r.shape
        elif r.shape[1] == 3:
            r, g, b = r[:, 0], r[:, 1], r[:, 2]
            original_shape = r.shape
        else:
            raise ValueError("入力が不正です。RGB画像または Nx3 配列を入力してください。")
    else:
        def normalize(x):
            if x.dtype == np.uint8:
                return x.astype(np.float64) / 255.0
            elif x.dtype == np.uint16:
                return x.astype(np.float64) / 65535.0
            elif np.max(x) > 1.0:
                return x.astype(np.float64) / 255.0
            return x.astype(np.float64)

        r, g, b = normalize(r), normalize(g), normalize(b)

        if not (r.shape == g.shape == b.shape):
            raise ValueError("r, g, b の形状が一致していません。")
        original_shape = r.shape

    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    denominator = 2 * r - g - b
    epsilon = 1e-10  # ゼロ除算回避
    h = np.arctan2((g - b), (denominator + epsilon))
    h[denominator == 0] = np.nan

    s = ((b - r)**2 + (r - g)**2 + (g - b)**2) / 3
    i = (r + g + b) / 3

    h = h.reshape(original_shape)
    s = s.reshape(original_shape)
    i = i.reshape(original_shape)

    return h, s, i


def compute_HD_raines(I, O):
    """
    入力画像 I と出力画像 O の Hue 成分誤差を、Raines の定義に基づき計算する関数。

    Parameters:
        I (ndarray): 入力RGB画像 (HxWx3, [0,255] または [0,1])
        O (ndarray): 出力RGB画像 (HxWx3, [0,255] または [0,1])

    Returns:
        float: Hue の平均誤差（複素平面上の距離）
    """
    I_h, _, _ = f_rgb2hsi_raines(I)
    O_h, _, _ = f_rgb2hsi_raines(O)
    diff = np.abs(np.exp(1j * I_h) - np.exp(1j * O_h))
    e = np.nanmean(diff)
    return e


def compute_HD_lab(I, O):
    """
    Lab 色空間における Hue 成分誤差を計算する関数。
    明度差および彩度差を除外して、Hue 差分を推定します。

    Parameters:
        I (ndarray): 入力RGB画像 (HxWx3, [0,255] または [0,1])
        O (ndarray): 出力RGB画像 (HxWx3, 同上)

    Returns:
        float: Hue 成分誤差の平均
    """
    if I.dtype != np.float64:
        I = I.astype(np.float64)
    if O.dtype != np.float64:
        O = O.astype(np.float64)
    if I.max() > 1.0:
        I = I / 255.0
    if O.max() > 1.0:
        O = O / 255.0

    I_lab = rgb2lab(I)
    O_lab = rgb2lab(O)

    I_sat = np.sqrt(I_lab[:, :, 1]**2 + I_lab[:, :, 2]**2)
    O_sat = np.sqrt(O_lab[:, :, 1]**2 + O_lab[:, :, 2]**2)

    E_diff = np.sqrt(np.sum((I_lab - O_lab)**2, axis=2))
    L_diff = np.abs(I_lab[:, :, 0] - O_lab[:, :, 0])
    sat_diff = I_sat - O_sat

    hue_diff = np.sqrt(np.clip(E_diff**2 - L_diff**2 - sat_diff**2, a_min=0, a_max=None))
    e = np.mean(np.abs(hue_diff))

    return e
