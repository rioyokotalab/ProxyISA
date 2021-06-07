from .base import *

class CUBirds(BaseDataset):
    def __init__(self, root, mode, scale, transform = None):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        self.scale = scale
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        if self.scale < 1.0:
            limit = 60
            capacity = [limit for x in range(10)]
            for c in range(1, 10):
                if limit * self.scale > 5:
                    limit *= self.scale
                else:
                    limit = 5
                capacity += [int(limit) for x in range(10)]
        #container = [0 for x in range(100)]
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                if self.scale < 1.0 and self.mode == 'train':
                    if capacity[y] > 0:
                        self.ys += [y]
                        self.I += [index]
                        self.im_paths.append(os.path.join(self.root, i[0]))
                        capacity[y] -= 1
                else:
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1
