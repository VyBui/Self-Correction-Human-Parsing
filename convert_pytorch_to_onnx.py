import argparse

import torch
import torchvision.transforms as transforms

import networks
from utils.transforms import BGR2RGB_transform

ONNX_FILE_PATH = 'self_correction_human_parsing.onnx'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert Pytorch to Onnx for"
                                                 "Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")

    parser.add_argument("--model-restore", type=str,
                        default='./log/checkpoint_108.pth.tar')
    parser.add_argument("--imagenet-pretrain", type=str,
                        default='./pretrain_model/resnet101-imagenet.pth')
    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)

    h, w = map(int, args.input_size.split(','))
    # input_size = [h, w]

    model = networks.init_model(args.arch, num_classes=args.num_classes,
                                pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    # if INPUT_SPACE == 'BGR':
    #     print('BGR Transformation')
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=IMAGE_MEAN,
    #                              std=IMAGE_STD),
    #
    #     ])
    # if INPUT_SPACE == 'RGB':
    #     print('RGB Transformation')
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         BGR2RGB_transform(),
    #         transforms.Normalize(mean=IMAGE_MEAN,
    #                              std=IMAGE_STD),
    #     ])

    x_input = torch.randn(1, 3, 512, 512, requires_grad=True)
    # Load model weight
    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.train(False)

    # x_input = torch.floatTensor(x_input)

    torch.onnx.export(model, x_input.cuda(), ONNX_FILE_PATH,
                      input_names=['input'],
                      output_names=['output'], export_params=True)


if __name__ == '__main__':
    main()
