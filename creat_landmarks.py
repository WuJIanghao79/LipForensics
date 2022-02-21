import argparse
import os
import time

from tqdm import tqdm
import numpy as np
from skimage import io
import face_alignment

DEVICE = 'cpu'
DATASETS = {
    "FaceForensics++": [
        "Forensics/RealFF",
        "Forensics/Deepfakes",
        "Forensics/FaceSwap",
        "Forensics/Face2Face",
        "Forensics/NeuralTextures",
    ],
    "RealFF": ["Forensics/RealFF"],
    "Deepfakes": ["Forensics/Deepfakes"],
    "FaceSwap": ["Forensics/FaceSwap"],
    "Face2Face": ["Forensics/Face2Face"],
    "NeuralTextures": ["Forensics/NeuralTextures"],
    "FaceShifter": ["Forensics/FaceShifter"],
    "DeeperForensics": ["Forensics/DeeperForensics"],
    "CelebDF": ["CelebDF/RealCelebDF", "CelebDF/FakeCelebDF"],
    "DFDC": ["DFDC"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="create landmarks via FAN")
    parser.add_argument("--data-root", help="Root path of datasets", type=str,
                        default='/Users/wujianghao/Desktop/LipForensics/data/datasets')
    parser.add_argument("--dataset", help="Dataset to preprocess", type=str,
                        choices=["all",
                                 "FaceForensics++",
                                 "RealFF",
                                 "Deepfakes",
                                 "FaceSwap",
                                 "Face2Face",
                                 "NeuralTextures",
                                 "FaceShifter",
                                 "DeeperForensics",
                                 "CelebDF",
                                 "DFDC", ],
                        default="RealFF", )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )

    args = parser.parse_args()

    return args


def save_landmarks(frame_path, frame_name, frame_root, video_name, arg,

                   method="FAN"
                   ):
    """

    Parameters
    ----------
    frame_path 帧画面路径
    frame_name 帧画面名称 0001.png
    frame_root 帧画面上一级路径
    video_name 视频名称
    method  处理方式， 默认FAN
    args

    Returns
    -------

    """
    global DEVICE,DATASETS
    save_path = os.path.join(args.data_root, DATASETS[args.dataset][0], args.compression, 'landmarks', video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    frame_name = frame_name[:-4]
    img = io.imread(frame_path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=DEVICE)
    frame_landmarks = fa.get_landmarks(img)[0]
    np.save(os.path.join(save_path, frame_name), frame_landmarks)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "all":
        dataset = ["Forensics/RealFF",
                   "Forensics/Deepfakes",
                   "Forensics/FaceSwap",
                   "Forensics/Face2Face",
                   "Forensics/NeuralTextures",
                   "Forensics/FaceShifter",
                   "Forensics/DeeperForensics", ]
    else:
        datasets = DATASETS[args.dataset]
    for dataset in datasets:
        compression = args.compression
        root = os.path.join(args.data_root, dataset, compression)  # data/datasets/Forensics/RealFF/c23
        video_root = os.path.join(root, "images")
        landmarks = os.path.join(root, "landmarks")

        video_folders = sorted(os.listdir(video_root))
        print(f"\n Processing {dataset}")
        for video in tqdm(video_folders):
            # 继续写这里
            landmarks_dir = os.path.join(landmarks, video)
            if not os.path.exists(landmarks):
                os.makedirs(landmarks, exist_ok=True)
            frame_root = os.path.join(video_root,
                                      video)  # data/datasets/Forensics/RealFF/c23/images/07__talking_against_wall

            for frame_name in sorted(os.listdir(frame_root)):
                frame_path = os.path.join(frame_root, frame_name)

                save_landmarks(frame_path, frame_name, frame_root, video, args)
