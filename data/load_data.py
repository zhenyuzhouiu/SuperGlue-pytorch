import os
import cv2
import math
import torch
import datetime
import numpy as np
import random
from models.superpoint_pytorch import SuperPoint
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def load_image(image_path, image_size, rgb=True):
    """

    Args:
        image_path:
        image_size: w x h
        rgb: Ture: rgb; False: gray

    Returns:
        dst_img: [h, w, c]

    """
    image = (
        cv2.imread(image_path, cv2.IMREAD_COLOR)
        if rgb
        else cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    )
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
        image = image[int(crop_h / 2) : int(h - crop_h / 2), :, :]
    else:
        crop_w = w - h / dst_r
        image = image[:, int(crop_w / 2) : int(w - crop_w / 2), :]

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
        corners = np.array(
            [[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32
        )
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(
            src=image, M=M, dsize=(image.shape[1], image.shape[0])
        )  # return an image type

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
                "keypoints0": torch.zeros([0, 0, 2], dtype=torch.double),
                "keypoints1": torch.zeros([0, 0, 2], dtype=torch.double),
                "descriptors0": torch.zeros([0, 2], dtype=torch.double),
                "descriptors1": torch.zeros([0, 2], dtype=torch.double),
                "image0": image,
                "image1": warped,
                "file_name": file_name,
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
        dists = cdist(kp1_projected, kp2_np)  # [n_kpts1, n_kpts2]

        # minimal indices of each column along row
        min1 = np.argmin(dists, axis=0)
        # minimal indices of each row along column
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate(
            [min1[matches][np.newaxis, :], matches[np.newaxis, :]]
        )  # mutual matched point
        # for the missing point, push them to the dust bin of sinkhorn
        MN2 = np.concatenate(
            [
                missing1[np.newaxis, :],
                (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64),
            ]
        )
        MN3 = np.concatenate(
            [
                (len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64),
                missing2[np.newaxis, :],
            ]
        )
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.0)
        descs2 = np.transpose(descs2 / 256.0)

        image = torch.from_numpy(image / 255.0).double()[None].cuda()
        warped = torch.from_numpy(warped / 255.0).double()[None].cuda()

        return {
            "keypoints0": list(kp1_np),
            "keypoints1": list(kp2_np),
            "descriptors0": list(descs1),
            "descriptors1": list(descs2),
            "scores0": list(scores1_np),
            "scores1": list(scores2_np),
            "image0": image,
            "image1": warped,
            "all_matches": list(all_matches),
            "file_name": file_name,
        }


class SuperPointDataset(Dataset):
    """Instead of using SIFT to extract keypoints and descriptors,
    it uses the SuperPoint to detect the keypoints and descriptors.
    """

    def __init__(
        self,
        train_path,
        sp_config,
        image_size,
        device,
        p_augment,
        transform=transforms.Compose([transforms.ToTensor()]),
    ):
        """
        image_size: [w, h]
        """
        self.files = []
        self.files += [os.path.join(train_path, f) for f in os.listdir(train_path)]
        superpoint = SuperPoint(**sp_config)
        self.superpoint = superpoint.to(device)
        self.device = device
        self.image_size = image_size
        self.p_augment = p_augment
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        p = np.random.rand(1)
        if p < self.p_augment:
            # RandomCrop
            p = np.random.rand(1)
            if p < 0.5:
                image_cv = Image.open(file_name)
                transform = transforms.RandomCrop(
                    (self.image_size[1], self.image_size[0]),
                    padding=None,
                    pad_if_needed=False,
                    fill=0,
                    padding_mode="constant",
                )
                image_cv = transform(image_cv)
            else:
                image_cv = load_image(file_name, image_size=self.image_size, rgb=True)
                image_cv = Image.fromarray(image_cv)

            # Random Rotation
            p = np.random.rand(1)
            if p < 0.3:
                transform = transforms.RandomRotation(degrees=(-15, 15))
                image_cv = transform(image_cv)

            # Gaussian Blur
            p = np.random.rand(1)
            if p < 0.3:
                transform = transforms.GaussianBlur(kernel_size=5)
                image_cv = transform(image_cv)

            # Color Jitter
            p = np.random.rand(1)
            if p < 0.5:
                transform = transforms.ColorJitter(
                    brightness=2, contrast=2, saturation=2, hue=0.5
                )
                image_cv = transform(image_cv)
        else:
            image_cv = load_image(file_name, image_size=self.image_size, rgb=True)
            image_cv = Image.fromarray(image_cv)

        # skip this image pair if no keypoints detected in image
        (
            kp1,
            descs1,
            score1,
            kp2,
            descs2,
            score2,
            image_pt,
            warped_pt,
            M,
        ) = self.sp_detection(image_cv)
        while kp1 is None:
            (
                kp1,
                descs1,
                score1,
                kp2,
                descs2,
                score2,
                image_pt,
                warped_pt,
                M,
            ) = self.sp_detection(image_cv)

        # convert tensor to numpy
        kp1 = kp1.data.cpu().numpy()
        kp2 = kp2.data.cpu().numpy()

        # obtain the matching matrix of the image pair by the keypoint location
        kp1_projected = cv2.perspectiveTransform(kp1.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2)  # [n_kp1, n_kp2]

        min1 = np.argmin(dists, axis=0)  # [n_kp2]
        min2 = np.argmin(dists, axis=1)  # [n_kp1]

        min1v = np.min(dists, axis=1)  # [n_kp1]
        min1f = min2[min1v < 3]  # depend (x, y) location

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]  # mutual nearest
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate(
            [
                missing1[np.newaxis, :],
                (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64),
            ]
        )
        MN3 = np.concatenate(
            [
                (len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64),
                missing2[np.newaxis, :],
            ]
        )
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        return {
            "keypoints0": kp1.reshape(-1, 2),
            "keypoints1": kp2.reshape(-1, 2),
            "descriptors0": descs1.transpose(1, 0).reshape(256, -1),
            "descriptors1": descs2.transpose(1, 0).reshape(256, -1),
            "scores0": score1.reshape(-1),
            "scores1": score2.reshape(-1),
            "image0": image_pt,
            "image1": warped_pt,
            "all_matches": all_matches.transpose((1, 0)),
            "file_name": file_name,
        }

    def sp_detection(self, image_cv):
        # augmentation for the warped_cv image
        p = np.random.rand(1)
        warped_cv = image_cv
        if p < self.p_augment:
            # Gaussian Blur
            p = np.random.rand(1)
            if p < 0.3:
                transform = transforms.GaussianBlur(kernel_size=5)
                warped_cv = transform(warped_cv)

            # Color Jitter
            p = np.random.rand(1)
            if p < 0.5:
                transform = transforms.ColorJitter(
                    brightness=2, contrast=2, saturation=2, hue=0.5
                )
                warped_cv = transform(warped_cv)

        # warp image
        transform = transforms.Grayscale(num_output_channels=1)
        image_cv = transform(image_cv)
        warped_cv = transform(warped_cv)
        image_cv = np.array(image_cv)
        warped_cv = np.array(warped_cv)
        width, height = image_cv.shape[:2]
        # height, width = image_cv.shape[:2]
        corners = np.array(
            [[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32
        )
        warp = np.random.randint(
            -int(min(self.image_size) / 2), int(min(self.image_size) / 2), size=(4, 2)
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped_cv = cv2.warpPerspective(
            src=warped_cv, M=M, dsize=(image_cv.shape[1], image_cv.shape[0])
        )  # return an image type

        # convert image from numpy to tensor
        image_pt, warped_pt = self.transform(image_cv), self.transform(warped_cv)
        image_pt, warped_pt = image_pt.to(self.device), warped_pt.to(self.device)
        pair = torch.cat([image_pt, warped_pt], dim=0).unsqueeze(1)  # [2, 1, 200, 152]

        # extract keypoints of the image pair using SuperPoint model under self.max_keypoints
        #    {"keypoints": keypoints,
        #     "keypoint_scores": scores,
        #     "descriptors": descriptors}
        pred = self.superpoint({"image": pair})
        kp1, descs1, score1 = (
            pred["keypoints"][0],
            pred["descriptors"][0],
            pred["scores"][0],
        )
        kp2, descs2, score2 = (
            pred["keypoints"][1],
            pred["descriptors"][1],
            pred["scores"][1],
        )
        if len(kp1) < 2 or len(kp2) < 2:
            return None, None, None, None, None, None, None, None, None
        return kp1, descs1, score1, kp2, descs2, score2, image_pt, warped_pt, M


class SPBatchDataset(Dataset):
    """Instead of using SIFT to extract keypoints and descriptors,
    it uses the SuperPoint to detect the keypoints and descriptors.

    Based on the SuperPointDataset(), it supports batch size (batch size > 0) training by padding zeros to form the same number.
    """

    def __init__(
        self,
        train_path,
        sp_config,
        image_size,
        device,
        transform=transforms.Compose([transforms.ToTensor()]),
    ):
        """
        image_size: [w, h]
        """
        self.sp_config = sp_config
        self.files = []
        self.files += [os.path.join(train_path, f) for f in os.listdir(train_path)]
        superpoint = SuperPoint(**sp_config)
        self.superpoint = superpoint.to(device)
        self.image_size = image_size
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        # image = opencv
        image_cv = load_image(file_name, image_size=self.image_size, rgb=False)

        # skip this image pair if no keypoints detected in image
        (
            kp1,
            descs1,
            score1,
            kp2,
            descs2,
            score2,
            image_pt,
            warped_pt,
            M,
        ) = self.sp_detection(image_cv)
        while kp1 is None:
            (
                kp1,
                descs1,
                score1,
                kp2,
                descs2,
                score2,
                image_pt,
                warped_pt,
                M,
            ) = self.sp_detection(image_cv)

        # convert tensor to nump
        kp1, descs1, score1 = (
            kp1.data.cpu().numpy(),
            descs1.data.cpu().numpy(),
            score1.data.cpu().numpy(),
        )
        kp2, descs2, score2 = (
            kp2.data.cpu().numpy(),
            descs2.data.cpu().numpy(),
            score2.data.cpu().numpy(),
        )

        # obtain the matching matrix of the image pair by the keypoint location
        kp1_projected = cv2.perspectiveTransform(kp1.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2)  # [n_kp1, n_kp2]
        min1 = np.argmin(dists, axis=0)  # [n_kp2]
        min2 = np.argmin(dists, axis=1)  # [n_kp1]
        min1v = np.min(dists, axis=1)  # [n_kp1]
        min1f = min2[min1v < 3]  # depend (x, y) location
        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]  # mutual nearest
        matches = np.intersect1d(min1f, xx)
        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])

        # for training with batch size, we should pad keypoint, desriptor, and score to the same length
        max_keypoints = self.sp_config["max_num_keypoints"]
        pad_kp1 = np.zeros((max_keypoints, 2))
        pad_kp1[: kp1.shape[0], :] = kp1
        pad_descs1 = np.zeros((max_keypoints, 256))
        pad_descs1[: descs1.shape[0], :] = descs1
        pad_score1 = np.zeros((max_keypoints,), dtype=score1.dtype)
        pad_score1[: score1.shape[0]] = score1
        pad_kp2 = np.zeros((max_keypoints, 2))
        pad_kp2[: kp2.shape[0], :] = kp2
        pad_descs2 = np.zeros((max_keypoints, 256))
        pad_descs2[: descs2.shape[0], :] = descs2
        pad_score2 = np.zeros((max_keypoints,), dtype=score2.dtype)
        pad_score2[: score2.shape[0]] = score2

        # obtain the missing matched keypoint

        missing1 = np.setdiff1d(np.arange(kp1.shape[0]), min1[matches])
        pad_missing1 = np.arange(kp1.shape[0], max_keypoints)
        pad_missing1 = np.concatenate([missing1, pad_missing1])
        MN2 = np.concatenate(
            [
                pad_missing1[np.newaxis, :],
                (len(pad_kp2)) * np.ones((1, len(pad_missing1)), dtype=np.int64),
            ]
        )
        missing2 = np.setdiff1d(np.arange(kp2.shape[0]), matches)
        pad_missing2 = np.arange(kp2.shape[0], max_keypoints)
        pad_missing2 = np.concatenate([missing2, pad_missing2])
        MN3 = np.concatenate(
            [
                (len(pad_kp1)) * np.ones((1, len(pad_missing2)), dtype=np.int64),
                pad_missing2[np.newaxis, :],
            ]
        )

        # matched + missing keypoint pair
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        # each batch all_matches size should also be same, for the worst case the all_matches should be [2, 512]
        pad_all_matches = all_matches
        dest_pair = max_keypoints * 2
        current_pair = all_matches.shape[1]
        matched_pair = MN.shape[1]
        if current_pair < dest_pair:
            diff_pair = dest_pair - current_pair
            if matched_pair >= diff_pair:
                pad_all_matches = np.concatenate(
                    [pad_all_matches, MN[:, :diff_pair]], axis=1
                )
            else:
                quotient = diff_pair // matched_pair
                remainder = diff_pair % matched_pair
                for q in quotient:
                    pad_all_matches = np.concatenate([pad_all_matches, MN], axis=1)
                if remainder != 0:
                    pad_all_matches = np.concatenate(
                        [pad_all_matches, MN[:, :remainder]], axis=1
                    )

        return {
            "keypoints0": pad_kp1.reshape(-1, 2),
            "keypoints1": pad_kp2.reshape(-1, 2),
            "descriptors0": pad_descs1.transpose(1, 0).reshape(256, -1),
            "descriptors1": pad_descs2.transpose(1, 0).reshape(256, -1),
            "scores0": pad_score1.reshape(-1),
            "scores1": pad_score2.reshape(-1),
            "image0": image_pt,
            "image1": warped_pt,
            "all_matches": pad_all_matches.transpose((1, 0)),
            "file_name": file_name,
        }

    def sp_detection(self, image_cv):
        # rigid warp image
        width, height = image_cv.shape[:2]
        corners = np.array(
            [[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32
        )
        warp = np.random.randint(-76, 76, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped_cv = cv2.warpPerspective(
            src=image_cv, M=M, dsize=(image_cv.shape[1], image_cv.shape[0])
        )  # return an image type

        # convert image from numpy to tensor
        image_pt, warped_pt = self.transform(image_cv), self.transform(warped_cv)
        image_pt, warped_pt = image_pt.to(self.device), warped_pt.to(self.device)

        pair = torch.cat([image_pt, warped_pt], dim=0).unsqueeze(1)  # [2, 1, 200, 152]

        # extract keypoints of the image pair using SuperPoint model under self.max_keypoints
        pred = self.superpoint({"image": pair})
        # kpt.shape= [n_kpt, 2], descs1.shape=[n_kpt, 2], score.shape=[n_kpt,]
        kp1, descs1, score1 = (
            pred["keypoints"][0],
            pred["descriptors"][0],
            pred["scores"][0],
        )  # [n_kp1, ]
        kp2, descs2, score2 = (
            pred["keypoints"][1],
            pred["descriptors"][1],
            pred["scores"][1],
        )  # [n_kp2, ]

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return None, None, None, None, None, None, None, None, None
        return kp1, descs1, score1, kp2, descs2, score2, image_pt, warped_pt, M


class SPPairDataset(Dataset):
    """Instead of using SIFT to extract keypoints and descriptors,
    it uses the SuperPoint to detect the keypoints and descriptors.

    Based on the SuperPointDataset(), it support:
    1. training image pairs from different samples of the same identity.
    
    2. adding keypoints from ground truth based on the detection results of SuperPoint? (The trained SuperPoint must detect keypoints that are exist on the ground truth)
    
    
    """

    def __init__(
        self,
        train_path,
        sp_config,
        image_size,
        device,
        p_augment, # probability of data augmentation
        p_sameid, # probability of the image pair from different samples of the same identity
        transform=transforms.Compose([transforms.ToTensor()]),
        no_sample=20
    ):
        """
        image_size: [w, h]
        """
        self.files = []
        self.files += [os.path.join(train_path, f) for f in os.listdir(train_path)]
        superpoint = SuperPoint(**sp_config)
        self.superpoint = superpoint.to(device)
        self.device = device
        self.image_size = image_size
        self.p_augment = p_augment
        self.p_sameid = p_sameid
        self.transform = transform
        self.no_sample=20

    def __len__(self):
        return len(self.files)
    
    def data_augment(self, file_name, if_sameid):
        p = np.random.rand(1)
        if p < self.p_augment:
            # RandomCrop
            p = np.random.rand(1)
            if p < 0.5 and if_sameid is False:
                image_cv = Image.open(file_name)
                transform = transforms.RandomCrop(
                    (self.image_size[1], self.image_size[0]),
                    padding=None,
                    pad_if_needed=False,
                    fill=0,
                    padding_mode="constant",
                )
                image_cv = transform(image_cv)
            else:
                image_cv = load_image(file_name, image_size=self.image_size, rgb=True)
                image_cv = Image.fromarray(image_cv)

            # Random Rotation
            p = np.random.rand(1)
            if p < 0.3 and if_sameid is False:
                transform = transforms.RandomRotation(degrees=(-15, 15))
                image_cv = transform(image_cv)

            # Gaussian Blur
            p = np.random.rand(1)
            if p < 0.3:
                transform = transforms.GaussianBlur(kernel_size=5)
                image_cv = transform(image_cv)

            # Color Jitter
            p = np.random.rand(1)
            if p < 0.5:
                transform = transforms.ColorJitter(
                    brightness=2, contrast=2, saturation=2, hue=0.5
                )
                image_cv = transform(image_cv)
        else:
            image_cv = load_image(file_name, image_size=self.image_size, rgb=True)
            image_cv = Image.fromarray(image_cv)
        return image_cv

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        file_name = self.files[idx]
        file_name2 = ''
        if_sameid = False
        image_cv, image_cv2 = None, None
        p = np.random.random(1)
        if p < self.p_sameid:
            id = int(os.path.basename(file_name).split('.')[0])
            id_i = id // self.no_sample # Because the 20 is sample number of DiffPalm dataset
            sample_i = id % self.no_sample
            sample_list = [i for i in range(20)]
            sample_list.remove(sample_i)
            sample_i = random.sample(sample_list)
            file_name2 = os.path.join(os.path.dirname(file_name), str(id_i * self.no_sample + sample_i) + '.jpg')
            if not os.path.exists(file_name2):
                file_name2 = ''
                if_sameid = False
            else:
                if_sameid = True
                image_cv2 = self.data_augment(file_name2, if_sameid)

        image_cv = self.data_augment(file_name, if_sameid)

        # skip this image pair if no keypoints detected in image
        (
            kp1,
            descs1,
            score1,
            kp2,
            descs2,
            score2,
            image_pt,
            warped_pt,
            M,
        ) = self.sp_detection(image_cv)
        while kp1 is None:
            (
                kp1,
                descs1,
                score1,
                kp2,
                descs2,
                score2,
                image_pt,
                warped_pt,
                M,
            ) = self.sp_detection(image_cv)

        # convert tensor to numpy
        kp1 = kp1.data.cpu().numpy()
        kp2 = kp2.data.cpu().numpy()

        # obtain the matching matrix of the image pair by the keypoint location
        kp1_projected = cv2.perspectiveTransform(kp1.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2)  # [n_kp1, n_kp2]

        min1 = np.argmin(dists, axis=0)  # [n_kp2]
        min2 = np.argmin(dists, axis=1)  # [n_kp1]

        min1v = np.min(dists, axis=1)  # [n_kp1]
        min1f = min2[min1v < 3]  # depend (x, y) location

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]  # mutual nearest
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate(
            [
                missing1[np.newaxis, :],
                (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64),
            ]
        )
        MN3 = np.concatenate(
            [
                (len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64),
                missing2[np.newaxis, :],
            ]
        )
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        return {
            "keypoints0": kp1.reshape(-1, 2),
            "keypoints1": kp2.reshape(-1, 2),
            "descriptors0": descs1.transpose(1, 0).reshape(256, -1),
            "descriptors1": descs2.transpose(1, 0).reshape(256, -1),
            "scores0": score1.reshape(-1),
            "scores1": score2.reshape(-1),
            "image0": image_pt,
            "image1": warped_pt,
            "all_matches": all_matches.transpose((1, 0)),
            "file_name": file_name,
        }

    def sp_detection(self, image_cv, image_cv2=None, if_sameid=False):
        """The trained SuperPoint is employed to detect the keypoint of image_cv and 

        Args:
            image_cv (_type_): _description_

        Returns:
            _type_: _description_
        """
        # augmentation for the warped_cv image
        p = np.random.rand(1)
        if if_sameid and image_cv2 is not None:
            warped_cv = image_cv2
        else:
            warped_cv = image_cv
        if p < self.p_augment:
            # Gaussian Blur
            p = np.random.rand(1)
            if p < 0.3:
                transform = transforms.GaussianBlur(kernel_size=5)
                warped_cv = transform(warped_cv)

            # Color Jitter
            p = np.random.rand(1)
            if p < 0.5:
                transform = transforms.ColorJitter(
                    brightness=2, contrast=2, saturation=2, hue=0.5
                )
                warped_cv = transform(warped_cv)

        # warp image
        transform = transforms.Grayscale(num_output_channels=1)
        image_cv = transform(image_cv)
        warped_cv = transform(warped_cv)
        image_cv = np.array(image_cv)
        warped_cv = np.array(warped_cv)
        width, height = image_cv.shape[:2]
        # height, width = image_cv.shape[:2]
        corners = np.array(
            [[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32
        )
        warp = np.random.randint(
            -int(min(self.image_size) / 2), int(min(self.image_size) / 2), size=(4, 2)
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped_cv = cv2.warpPerspective(
            src=warped_cv, M=M, dsize=(image_cv.shape[1], image_cv.shape[0])
        )  # return an image type

        # convert image from numpy to tensor
        image_pt, warped_pt = self.transform(image_cv), self.transform(warped_cv)
        image_pt, warped_pt = image_pt.to(self.device), warped_pt.to(self.device)
        pair = torch.cat([image_pt, warped_pt], dim=0).unsqueeze(1)  # [2, 1, 200, 152]

        # extract keypoints of the image pair using SuperPoint model under self.max_keypoints
        #    {"keypoints": keypoints,
        #     "keypoint_scores": scores,
        #     "descriptors": descriptors}
        pred = self.superpoint({"image": pair})
        kp1, descs1, score1 = (
            pred["keypoints"][0],
            pred["descriptors"][0],
            pred["scores"][0],
        )
        kp2, descs2, score2 = (
            pred["keypoints"][1],
            pred["descriptors"][1],
            pred["scores"][1],
        )
        if len(kp1) < 2 or len(kp2) < 2:
            return None, None, None, None, None, None, None, None, None
        return kp1, descs1, score1, kp2, descs2, score2, image_pt, warped_pt, M
