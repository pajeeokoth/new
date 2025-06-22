from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import predict
import numpy as np
import cv2
import io
import torch
import network
from torchvision import transforms as T
from datasets import VOCSegmentation, Cityscapes

app = FastAPI()

# --- Model Setup (adjust as needed) ---
MODEL_NAME = 'deeplabv3plus_mobilenet'
NUM_CLASSES = 19  # or 19 for cityscapes (was 21)
DECODE_FN = VOCSegmentation.decode_target  # or Cityscapes.decode_target
CKPT_PATH = './best_deeplabv3plus_mobilenet_cityscapes_os16.pth'#'path/to/your/checkpoint.pth'  # update this

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=16)
checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# ##########
# @app.route('/')
# def index():
#     print('Request for index page received')
#     return render_template('index.html')
# ##########

# --- Image Transformation ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha=0.3) -> Image.Image:
    mask = mask.convert("RGBA")
    image = image.convert("RGBA")
    blended = Image.blend(image, mask, alpha)
    return blended

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]
        color_mask = DECODE_FN(pred).astype('uint8')
        mask_img = Image.fromarray(color_mask).resize(img.size)

    overlayed = overlay_mask_on_image(img, mask_img, alpha=0.5)
    buf = io.BytesIO()
    overlayed.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
