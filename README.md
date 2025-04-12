# 画像品質評価ツール / Code for Running Quantitative Image Processing Metrics
画像処理アルゴリズムの性能を、定量的に評価するためのPythonツールです。複数フォルダにわたる評価画像を一括処理し、定量評価指標を計算・可視化します。
- 複数フォルダを一括評価 & 結果をCSV出力
- 箱ひげ図による視覚的な比較を自動生成
- 評価指標は柔軟に選択可能

## 評価指標一覧 / Supported Metrics

| 指標名     | 内容                         |
|------------|------------------------------|
| PSNR       | ピーク信号対雑音比           |
| SSIM       | 構造的類似度                 |
| UIQM       | 水中画像の品質評価指標             |
| UCIQE      | 水中画像の品質評価指標      　　　 |
| std_L      | L\*a\*b*色空間におけるL\*の標準偏差 （コントラスト評価）              |
| CR         | 強調処理後に白飛びや黒潰れを引き起こした画素の割合 |
| HD_Raines  | Rainesの色相における色相差           |
| HD_Lab     | Lab空間の色相における色相差            |

## フォルダ構成 / Directory Structure

```bash
main/
├── src/
│   ├── image_metrics.py/
├── LICENSE      
├── README.md/
├── main.py/
├── requirements.txt/
```

## 実行環境 / Execution environment

以下の環境で動作確認を行いました：

- OS: Windows 11
- Python: 3.11.11
- 仮想環境: conda

必要なライブラリは `requirements.txt` を使ってインストールしてください：

```bash
pip install -r requirements.txt
```

## ライセンス / License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.  
Developed by h-mori2002, k-nanaura

This project references code from the following repository:

- [PSNR-SSIM-UCIQE-UIQM-Python](https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python) by Xuelei Chen, licensed under the MIT License.
