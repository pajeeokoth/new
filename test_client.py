import requests

# Update this path to the image you want to test
image_path = '../data/leftimg8bit/train/cologne/cologne_000129_000019_leftImg8bit.png' #"test_image.jpg"
# The FastAPI server URL
url = "http://127.0.0.1:8000/segment/"

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open("overlayed_result.png", "wb") as out:
        out.write(response.content)
    print("Overlayed image saved as overlayed_result.png")
else:
    print("Request failed:", response.status_code, response.text)