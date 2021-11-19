import os

ROOT_DIR = os.path.abspath("./")
DEFAULT_TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "data/train")
DEFAULT_PREVIOUS_WEIGHT_PATH = os.path.join(ROOT_DIR, "weight/weight.ckpt.0")

from model.loss.focal_loss import categorical_focal_loss


def train(
        model, train_gen, epochs, step, num_ckpt_to_save, ckpt_path, load_weights
):
    model.compile(loss=categorical_focal_loss(gamma=2., alpha=.25), optimizer='adadelta')

    if os.path.exists(load_weights):
        print("load weight successfully!")
        model.load_weights(load_weights)

    for ep in range(int(epochs / num_ckpt_to_save)):
        print("Starting Epoch ", ep)
        model.fit_generator(train_gen, steps_per_epoch=step, epochs=num_ckpt_to_save, use_multiprocessing=False,
                            workers=2, max_queue_size=2)
        if not ckpt_path is None:
            model.save_weights(ckpt_path + "." + str(ep))
            print("saved ", ckpt_path + "." + str(ep))
        print("Finished Epoch", ep)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train for image segmentation',
    )
    parser.add_argument(
        '--train_dir', required=False, default=DEFAULT_TRAIN_DATA_DIR,
        metavar="/path/to/trainDir/", help='data directory for training'
    )
    parser.add_argument(
        '--categories', required=False, default=2, metavar=2, help="categories", type=int,
    )
    parser.add_argument(
        '--prev_weight', required=False, default=DEFAULT_PREVIOUS_WEIGHT_PATH,
        metavar="/path/to/prev_weight.ckpt", help="model weight"
    )
    parser.add_argument(
        '--input_length', required=False, default=512, metavar=512, help="input_length", type=int,
    )

    parser.add_argument(
        '--input_width', required=False, default=512, metavar=512, help="input_width", type=int,
    )

    parser.add_argument(
        '--epochs', required=False, default=10, metavar=10, help="epochs", type=int,
    )

    parser.add_argument(
        '--batch_size', required=False, default=1, metavar=2, help="batch_size", type=int,
    )

    args = parser.parse_args()
    print("train_dir: ", args.train_dir)
    print("categories: ", args.categories)
    print("prev_weight: ", args.prev_weight)
    print("input_length: ", args.input_length)
    print("input_width: ", args.input_width)
    print("batch_size: ", args.batch_size)
    print("epochs: ", args.epochs)

    input_size = (args.input_length, args.input_width)
    n_classes = args.categories

    from model.architecture.unet import unet
    from model.backbone.resnet import get_resnet50_encoder
    from utils.data_loader import image_segmentation_generator

    model, output_size = unet(args.categories, get_resnet50_encoder, input_size=input_size)

    train_img_dir = os.path.join(args.train_dir, 'img')
    train_mask_dir = os.path.join(args.train_dir, 'mask')

    ckpt_path = os.path.join(ROOT_DIR, "weight/weight.ckpt")
    print(ckpt_path)
    train_gen, step = image_segmentation_generator(
        train_img_dir, train_mask_dir, args.batch_size, n_classes, input_size, output_size
    )

    train(
        model, train_gen, args.epochs, step, 1, ckpt_path, args.prev_weight
    )
