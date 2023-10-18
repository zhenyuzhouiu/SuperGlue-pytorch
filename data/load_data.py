import numpy as np
import torch
import os
import cv2
import math
import datetime

from models.superpoint_pytorch import SuperPoint
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from torchvision import transforms


def load_image(image_path, image_size, rgb=True):
    """

    Args:
        image_path:
        image_size: w x h
        rgb: Ture: rgb; False: gray

    Returns:
        dst_img: [h, w, c]

    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if rgb else cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    h, w, c = image.shape
    r = h / w
    dst_w, dst_h = image_size
    dst_r = dst_h / dst_w
    if r > dst_r:  # crop h
        crop_h = h - dst_r * w
        image = image[int(crop_h / 2):int(h - crop_h / 2), :, :]
    else:
        crop_w = w - h / dst_r
        image = image[:, int(crop_w / 2):int(w - crop_w / 2), :]

    dst_image = cv2.resize(image, dsize=(dst_w, dst_h))
    # dst_image = np.expand_dims(dst_image, -1) if dst_image.ndim == 2 else dst_image

    return dst_image


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures):
        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        # self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        sift = self.sift
        width, height = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))  # return an image type

        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            }

            # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])
        scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        # obtain the matching matrix of the image pair
        matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        image = torch.from_numpy(image / 255.).double()[None].cuda()
        warped = torch.from_numpy(warped / 255.).double()[None].cuda()

        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        }


class SuperPointDataset(Dataset):
    """ Instead of using SIFT to extract keypoints and descriptors,
    it uses the SuperPoint to detect the keypoints and descriptors.
    """

    def __init__(self, train_path, sp_config, image_size,
                 transform=transforms.Compose([transforms.ToTensor()])):
        """
        image_size: [w, h]
        """
        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]
        self.superpoint = SuperPoint(**sp_config)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = load_image(file_name, image_size=self.image_size, rgb=False)

        # warp image
        width, height = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))  # return an image type

        # convert image from numpy to tensor
        image, warped = self.transform(image), self.transform(warped)
        pair = torch.cat([image, warped], dim=0).unsqueeze(1)  # [2, 1, 200, 152]

        # extract keypoints of the image pair using SuperPoint model under self.max_keypoints
        #    {"keypoints": keypoints,
        #     "keypoint_scores": scores,
        #     "descriptors": descriptors}
        pred = self.superpoint({'image': pair})
        kp1, descs1, score1 = pred["keypoints"][0], pred["descriptors"][0], pred["keypoint_scores"][0]
        kp2, descs2, score2 = pred["keypoints"][1], pred["descriptors"][1], pred["keypoint_scores"][1]

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            }
        # convert tensor to numpy
        kp1 =

        # obtain the matching matrix of the image pair by the keypoint location
        kp1_projected = cv2.perspectiveTransform(kp1.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        image = torch.from_numpy(image / 255.).double()[None].cuda()
        warped = torch.from_numpy(warped / 255.).double()[None].cuda()

        return {
            'keypoints0': list(kp1),
            'keypoints1': list(kp2),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(score1),
            'scores1': list(score2),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        }
