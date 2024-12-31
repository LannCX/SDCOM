from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataloderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataloderX, self).__iter__())
