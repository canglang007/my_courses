import random
import numbers
import math
import collections

from PIL import ImageOps, Image
import numpy as np

# 填充
class Padding:
    def __init__(self, pad):
        self.pad = pad

    # call方法使得类可以直接相当于函数的方式来调用
    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)

# 调整尺寸
class Scale:
    # 参数是尺寸和插值方法，Image.NEAREST是最邻近插值法
    def __init__(self, size, interpolation=Image.NEAREST):
        # 判断size是int型或是是一个长度为2的可迭代对象
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        # 如果输入的size是一个int类型的整数的话
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, target
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation), target.resize((ow, oh), self.interpolation)
            # 如果w>h
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation), target.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation), target.resize(self.size, self.interpolation)

# 中心裁剪
class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img, target = imgmap
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))

# 随机裁剪
class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img, target = imgmap
        w, h = img.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return img, target
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            return img.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img, target

# 随机大小的裁剪
class RandomSizedCrop:

    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                target = target.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(target.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation), \
                       target.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale((img, target)))


class RandomHorizontalFlip:

    def __call__(self, imgmap):
        img, target = imgmap
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        return img, target


class RandomRotation:

    def __call__(self, imgmap, degree=10):
        img, target = imgmap
        # 一个随机的角度
        deg = np.random.randint(-degree, degree, 1)[0]
        return img.rotate(deg), target.rotate(deg)
