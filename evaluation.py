import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
#import eva
from util.image import unnormalize

def evaluate(model, dataset, device, filename):
    #change i to generate different images
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask) 
    gt = torch.stack(gt) #ground truth
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))# image gained by network
      
    output_comp = mask * image + (1 - mask) * output
    #generate images to compare, optional
    grid = make_grid(
        torch.cat((unnormalize(image), unnormalize(output),
                   unnormalize(gt)), dim=0))
    save_image(grid, './result/output/test_{:d}.jpg'.format(i))
    #save discrete images
    save_image(unnormalize(output), './result/output/test_{:d}.jpg'.format(i))
    save_image(unnormalize(gt), './result/gt/test_{:d}.jpg'.format(i)) 
