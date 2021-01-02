import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
#import eva
from util.image import unnormalize

def evaluate(model, dataset, device, filename):
    for i in range(3000):
        image, mask, gt = zip(*[dataset[i]])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)#原始图像
        with torch.no_grad():
           output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))#训练后的图像
        #待比较的应该是output和gt
        output_comp = mask * image + (1 - mask) * output
        '''grid = make_grid(
            torch.cat((unnormalize(image), unnormalize(output),
                       unnormalize(gt)), dim=0))
        save_image(grid, './result/output/test_{:d}.jpg'.format(i))'''
        save_image(unnormalize(output), './result/output/test_{:d}.jpg'.format(i))
        save_image(unnormalize(gt), './result/gt/test_{:d}.jpg'.format(i)) 
