import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Image processing branch (ResNet50)
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Freeze feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Image branch fully connected layers
        self.fc_image = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images):
        # Process images through ResNet
        x = self.base_model(images)
        x = torch.flatten(x, 1)
        image_features = self.fc_image(x)
        
        # Final classification
        output = self.classifier(image_features)
        return output


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, num_landmarks=21):
        super().__init__()
       
        # Image processing branch (ResNet50)
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
       
        # Freeze feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = False
           
        # Image branch fully connected layers
        self.fc_image = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
       
        # Landmark processing branch
        self.fc_landmarks = nn.Sequential(
            nn.Linear(num_landmarks * 2, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
       
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
       
    def forward(self, images, landmarks):
        # Process images through ResNet
        x = self.base_model(images)
        x = torch.flatten(x, 1)
        image_features = self.fc_image(x)
       
        # Process landmark features
        batch_size = landmarks.size(0)
        landmarks_flat = landmarks.view(batch_size, -1)
        landmark_features = self.fc_landmarks(landmarks_flat)
       
        # Combine features
        combined = torch.cat((image_features, landmark_features), dim=1)
       
        # Final classification
        output = self.classifier(combined)
        return output