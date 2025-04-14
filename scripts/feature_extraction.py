from torchvision import transforms
from PIL import Image
import torch

# Define the transformations for preprocessing the input image
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(cropped_image, model):
    """
    Extracts feature from the given cropped image using the pretrained Re-ID model.
    """
    # Convert the image to a PIL format and apply the transformations
    person_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    img = transform(person_pil).unsqueeze(0).cuda()  # Move to GPU if available
    
    with torch.no_grad():
        features = model(img)  # Extract features from the model
    
    return features.cpu().numpy()  # Convert tensor to numpy array
