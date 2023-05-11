# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/5/1 21:49
from torch.utils.data import Dataset, DataLoader
import pickle
from torchvision.transforms import transforms
from PIL import Image

IMAGE = 'IMAGE'
VIBRATION = 'VIBRATION'
LABEL = 'LABEL'
TRAIN = 'Train'
VALID = 'Valid'
TEST = 'Test'


def total(params):
    """
    count the total number of hyperparameter settings
    """
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def load_Rail(data_path):
    # parse the input args

    class Rail(Dataset):
        '''
        PyTorch Dataset for POM, don't need to change this
        '''

        def __init__(self, image, vibration, labels, img_transform=None, vibration_transform=None):
            self.image = image
            self.vibration = vibration
            self.labels = labels
            self.img_trans = img_transform
            self.vibration_trans = vibration_transform

        def __getitem__(self, idx):
            img = self.image[idx]
            vib = self.vibration[idx]
            label = self.labels[idx]
            if self.img_trans is not None:
                img = self.img_trans(img)
            if self.vibration_trans is not None:
                vib = self.vibration_trans(vib)
            return [img, vib, label]

        def __len__(self):
            return len(self.image)

    Rail_data = pickle.load(open(data_path + "Rail_dataset_366.pkl", 'rb'))

    Rail_train, Rail_valid, Rail_test = Rail_data[TRAIN], Rail_data[VALID], Rail_data[TEST]

    train_img, train_vib, train_labels = Rail_train[IMAGE], Rail_train[VIBRATION], Rail_train[LABEL]
    valid_img, valid_vib, valid_labels = Rail_valid[IMAGE], Rail_valid[VIBRATION], Rail_valid[LABEL]
    test_img, test_vib, test_labels = Rail_test[IMAGE], Rail_test[VIBRATION], Rail_test[LABEL]

    Image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    VIBRATION_transform = transforms.ToTensor()

    # code that instantiates the Dataset objects
    train_set = Rail(train_img, train_vib, train_labels, img_transform=Image_transform,
                     vibration_transform=VIBRATION_transform)
    valid_set = Rail(valid_img, valid_vib, valid_labels, img_transform=Image_transform,
                     vibration_transform=VIBRATION_transform)
    test_set = Rail(test_img, test_vib, test_labels, img_transform=Image_transform,
                    vibration_transform=VIBRATION_transform)

    # train_set = Rail(train_img, train_vib, train_labels)
    # valid_set = Rail(valid_img, valid_vib, valid_labels)
    # test_set = Rail(test_img, test_vib, test_labels)

    img_dim = train_set[0][0].shape[0]
    print("Image feature dimension is: {}".format(img_dim))
    vib_dim = train_set[0][1].shape[2]
    print("Vibration feature dimension is: {}".format(vib_dim))

    input_dims = (img_dim, vib_dim)

    return train_set, valid_set, test_set, input_dims
#
#
# root = 'D:\\Pytorch\\Fusion_lowrank\\data\\'
#
# a, b, c, d = load_Rail(root)
# print('================================')
