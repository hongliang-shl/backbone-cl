from PIL import Image
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from facenet import Facenet

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model  # 要进行Grad-CAM处理的模型
        self.target_layer = target_layer  # 要进行特征可视化的目标层
        self.feature_maps = None  # 存储特征图
        self.gradients = None  # 存储梯度
        
        # 为目标层添加钩子，以保存输出和梯度
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)
 
    def save_feature_maps(self, module, input, output):
        """保存特征图"""
        self.feature_maps = output.detach()
 
    def save_gradients(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
 
    def generate_cam(self, image, class_idx=None):
        """生成CAM热力图"""
        # 将模型设置为评估模式
        self.model.eval()
        #torch.autograd.set_detect_anomaly(True)
        
        # 正向传播
        output = self.model(image)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
 
        # 清空所有梯度
        self.model.zero_grad()
 
        # 对目标类进行反向传播
        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot.cuda(), retain_graph=True)
 
        # 获取平均梯度和特征图
        print(self.gradients.shape)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.feature_maps.squeeze(0)
        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]
        
        # 创建热力图
        heatmap = torch.mean(activation, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(torch.from_numpy(heatmap))
        heatmap = cv2.resize(heatmap.numpy(), (image.size(3), image.size(2)))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 将热力图叠加到原始图像上
        original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return heatmap, superimposed_img
 
    def unprocess_image(self, image):
        """反预处理图像，将其转回原始图像"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((image.transpose(1, 2, 0) * std) + mean) * 255).astype(np.uint8)
        return image
 
def visualize_gradcam(model, input_image_path, target_layer):
    """可视化Grad-CAM热力图"""
    # 加载图像
    img = Image.open(input_image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).cuda()
 
    # 创建GradCAM
    gradcam = GradCAM(model, target_layer)
    heatmap, result = gradcam.generate_cam(input_tensor)
 
    # 显示图像和热力图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(heatmap)
    plt.title('heatmap')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(result)
    plt.title('overlay img')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model = Facenet(modelpath='/home/user/hongliang.shl/dev/coinAI-main/algorithm/coinrecogination/facenet-pytorch/modelfromdongqing/ep264-loss0.006-val_loss0.066.pth')
    #visualize_gradcam(model, "path_to_your_input_image.jpg", model.layer3[-1])
    image_1 = input('Input image_1 filename:')
    visualize_gradcam(model.net, image_1, model.net.backbone.model.block8)
    '''
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        probability = model.detect_image(image_1,image_2)
        print(probability)
    '''
