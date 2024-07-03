import os
import torch
from torch.utils.data import DataLoader
from mydataset import MyDataset
from network.model import Panoformer as PanoBiT
from PIL import Image
import numpy as np

class Tester:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        test_dataset = MyDataset('/notebooks/ue5','/notebooks/ue5/test.txt', is_training=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.settings.num_workers, pin_memory=True)
                                     
        self.model = PanoBiT()
        self.model.to(self.device)

        assert self.settings.load_weights_dir is not None, "Please provide the path to the trained weights"
        self.load_model()
        
        self.model.eval()

        os.makedirs(self.settings.output_dir, exist_ok=True)

    def load_model(self):
        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def test(self):
        with torch.no_grad():
            for inputs in self.test_loader:
                equi_inputs = inputs["normalized_rgb"].to(self.device)
                outputs = self.model(equi_inputs)
                pred_depth = outputs["pred_depth"].detach().cpu().numpy()[0,0]
                
                # Get the original filename from the dataset
                filename = os.path.basename(inputs["filename_depth"][0])
                
                # Save the predicted depth map as a 16-bit PNG image with the corresponding name
                depth_image = (pred_depth * 65535).astype(np.uint16)
                depth_image = Image.fromarray(depth_image)
                print(inputs["filename_rgb"][0])
                print(os.path.join(self.settings.output_dir, filename))
                depth_image.save(os.path.join(self.settings.output_dir, filename))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Panoformer testing script')
    parser.add_argument('--load_weights_dir', type=str, required=True, help='path to the trained weights')
    parser.add_argument('--gpu_devices', nargs='+', type=int, default=[0], help='the GPU devices to use')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='/notebooks/ue5/pred', help='directory to save the predicted depth maps')

    settings = parser.parse_args()
    tester = Tester(settings)
    tester.test()