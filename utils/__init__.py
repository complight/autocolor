import torch
import odak


class dataset(torch.utils.data.Dataset):
    def __init__(
                 self, 
                 directory, 
                 filename = None, 
                 key = '*.pt', 
                 device = torch.device('cpu')
                ):
        super(dataset, self).__init__()
        self.device = device
        if isinstance(filename, type(None)):
            self.input_data = odak.tools.list_files(directory, key = key)
        else:
            self.input_data = [filename, ]
        m = len(self.input_data)
        if len(self.input_data) > m:
            self.input_data = self.input_data[0:m]

   
    def __getitem__(self, index):
        if self.input_data[index][-3::] == 'png':
            image = odak.learn.tools.load_image(self.input_data[index], normalizeby = 255., torch_style = True)[0:3]
            laser_powers = torch.ones(1, 3, 3)
            return image, laser_powers
        data = odak.learn.tools.torch_load(self.input_data[index])
        image = data['target']
        laser_powers = data['laser powers']
        return image, laser_powers.unsqueeze(0)


    def __len__(self):
        return len(self.input_data)
