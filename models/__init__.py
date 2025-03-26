import importlib
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_detector_model(args):
    model_module = importlib.import_module("models.detector_models." + args.model)

    model_name_map = {
        "cnn_bigru_selfattention": "CNN2D_BiGRU"
    }

    model_class_name = model_name_map.get(args.model)
    if not model_class_name:
        raise ValueError(f"Model '{args.model}' is not in model_name_map")

    model_class = getattr(model_module, model_class_name)
    return model_class


def grad_cam(args, model, data):
    target_layer = model.classifier[-1]
    input_tensor = data

    use_cuda = not args.cpu
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)

    target_category = args.output_dim

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
