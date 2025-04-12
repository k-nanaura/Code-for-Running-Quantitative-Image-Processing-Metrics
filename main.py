import numpy as np
from skimage import io
import os
import csv
from src.image_metrics import rmetrics, nmetrics, compute_pcqi, compute_cr, compute_HD_raines, compute_HD_lab, plot_boxplots

###########################
# メイン関数
###########################
def main():
    """
    複数のフォルダをまとめて処理し、UIQM, UCIQE の平均・標準偏差（不偏）に加え、
    参照画像との PSNR/SSIM/PCQI も算出する。

    ※参照画像はフォルダ内の順番で対応付ける。
    """
    # 計算したい指標をリストで指定
    selected_metrics = ["PSNR", "SSIM", "UIQM", "UCIQE", "PCQI", "std_L", "CR", "HD_Raines", "HD_Lab"]  # HDを追加
    
    # 評価指標に対応する関数
    metrics_functions = {
        "PSNR": lambda ref, img: rmetrics(ref, img)[0],
        "SSIM": lambda ref, img: rmetrics(ref, img)[1],
        "UIQM": lambda ref, img: nmetrics(img)[0],
        "UCIQE": lambda ref, img: nmetrics(img)[1],
        "std_L": lambda ref, img: nmetrics(img)[2],
        "PCQI": lambda ref, img: compute_pcqi(ref, img),
        "CR": lambda raw, img: compute_cr(raw, img),  
        "HD_Raines": lambda raw, img: compute_HD_raines(raw, img),
        "HD_Lab": lambda raw, img: compute_HD_lab(raw, img) 
    }
    
    # 参照画像のフォルダパス（必要に応じて変更）
    ref_folder = Please write the path of the reference image folder here
    # 原画像のフォルダパス（必要に応じて変更）
    raw_folder = Please write the path of the raw image folder here
    
    # 評価対象のフォルダリスト（必要に応じて変更）
    folder_list = [
        Please write the path of the evaluation image folder here
    ]

    # 出力CSVファイル名
    output_csv = 'metrics_results.csv'

    # 箱ひげ図の保存先ディレクトリ
    boxplot_save_dir = r'C:\Users\morihiroki\Desktop\mori\results\plots'

     # CSVに書くヘッダ行
    csv_header = ['Folder'] + [f"Mean_{m}" for m in selected_metrics] + [f"Std_{m}" for m in selected_metrics]
    
    # 結果を一時的に格納するリスト
    all_results = []

    # 結果を格納する辞書
    results_dict = {}

    # 参照フォルダ内の画像を取得・ソート
    ref_images = sorted([f for f in os.listdir(ref_folder) if f.lower().endswith(('.png', '.jpg'))])
    # 原画像フォルダ内の画像を取得・ソート
    raw_images = sorted([f for f in os.listdir(raw_folder) if f.lower().endswith(('.png', '.jpg'))])

    # 各フォルダを順に処理
    for folder_idx, folder_path in enumerate(folder_list, start=1):
        if not os.path.isdir(folder_path):
            print(f'Warning: {folder_path} は有効なディレクトリではありません。スキップします。')
            all_results.append([folder_path] + [None] * (2 * len(selected_metrics)))
            continue

        # フォルダ内の png/jpg 画像を探索・ソート
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])

        if len(image_files) == 0:
            print(f'{folder_path} 内に png/jpg 画像が見つかりません。')
            all_results.append([folder_path] + [None] * (2 * len(selected_metrics)))
            continue

        # 参照画像と対象画像の数を比較し、最小の長さに合わせる
        min_len = min(len(ref_images), len(image_files), len(raw_images))

        metric_values = {m: [] for m in selected_metrics}

        for i in range(min_len):
            img_path = os.path.join(folder_path, image_files[i])
            ref_img_path = os.path.join(ref_folder, ref_images[i])
            raw_img_path = os.path.join(raw_folder, raw_images[i])  # 原画像のパス

            # 画像の読み込み
            img = io.imread(img_path)
            ref_img = io.imread(ref_img_path)
            raw_img = io.imread(raw_img_path)  # 原画像の読み込み
            
            # 選択した評価指標の算出
            for metric in selected_metrics:
                # CRのみ原画像を用いる。それ以外は従来通り参照画像を用いる
                if metric == "CR":
                    metric_val = metrics_functions[metric](raw_img, img)
                else:
                    metric_val = metrics_functions[metric](ref_img, img)
                metric_values[metric].append(metric_val)
                
            # 進捗表示
            msg = f'[フォルダ {folder_idx}/{len(folder_list)}]{image_files[i]}: ' + ", ".join([f"{metric}={metric_values[metric][-1]:.4f}" for metric in selected_metrics])
            print(msg)

        results_dict[folder_path] = {metric: metric_values[metric] for metric in selected_metrics}

        # 選択した評価指標の平均・不偏標準偏差の計算
        all_results.append([
            folder_path,
            *[np.mean(metric_values[m]) for m in selected_metrics],
            *[np.std(metric_values[m], ddof=1) for m in selected_metrics]
        ])

    # 結果を CSV ファイルに出力
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        for row in all_results:
            writer.writerow(row)

    print(f'\n完了しました。結果は {output_csv} に保存されました。')

    # 箱ひげ図作成
    plot_boxplots(results_dict, folder_list, selected_metrics, x_labels='提案手法', save_dir=boxplot_save_dir)

if __name__ == '__main__':
    main()
