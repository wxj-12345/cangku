import torch
import torchvision.transforms as T
import hashlib

INFER_COUNT = 0
pred_cache = {}
blur_op = T.GaussianBlur(kernel_size=(7, 7), sigma=(1.2, 1.8))

def get_infer_count():
    return INFER_COUNT

def reset_infer_count():
    global INFER_COUNT, pred_cache
    INFER_COUNT = 0
    pred_cache.clear()

def defend_preprocess(imgs):
    return blur_op(imgs)

def get_tensor_md5(tensor: torch.Tensor) -> str:
    byte_data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.md5(byte_data).hexdigest()

def model_predict(model, images):
    global INFER_COUNT
    batch_out = []
    model.eval()
    with torch.no_grad():
        for single_img in images:
            img_hash = get_tensor_md5(single_img)
            if img_hash in pred_cache:
                batch_out.append(pred_cache[img_hash])
                continue
            proc_img = blur_op(single_img.unsqueeze(0))
            logits = model(proc_img)
            pred = torch.argmax(logits, dim=1)[0]
            pred_cache[img_hash] = pred
            INFER_COUNT += 1
            batch_out.append(pred)
    return torch.stack(batch_out)

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.predict_counter = 0