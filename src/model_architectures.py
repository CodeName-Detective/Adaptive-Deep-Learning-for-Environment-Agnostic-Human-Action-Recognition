import torch
import numpy as np

class AlexNet3D(torch.nn.Module):
    
    def __init__(self: 'AlexNet3D', input_size: tuple, output_size: int)-> None:
        """Function to initate the model layers

        Args:
            input_size (tuple): Input size of the video as [dilation, height, width]
            
            output_size (int): Total number of output classes
        """
        super(AlexNet3D,self).__init__()
    
        self.max_pool = torch.nn.MaxPool3d(kernel_size=3, padding=(1,1,1),stride=2)
        
        self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=96, padding=(3,0,0), kernel_size=5, stride=2)
        conv1_size = self._get_output_size(input_size, self.conv1.padding, self.conv1.dilation, self.conv1.stride, self.conv1.kernel_size)
        pool1_size = self._get_output_size(conv1_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        self.conv2 = torch.nn.Conv3d(in_channels=96, out_channels=256, kernel_size=5, padding=(3,2,2))
        conv2_size = self._get_output_size(pool1_size, self.conv2.padding, self.conv2.dilation, self.conv2.stride, self.conv2.kernel_size)
        pool2_size = self._get_output_size(conv2_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.conv3 = torch.nn.Conv3d(in_channels=256, out_channels=384, kernel_size=3, padding=(3,1,1))
        conv3_size = self._get_output_size(pool2_size, self.conv3.padding, self.conv3.dilation,self.conv3.stride, self.conv3.kernel_size)
        
        self.conv4 = torch.nn.Conv3d(in_channels=384, out_channels=384, kernel_size=3, padding=(3,1,1))
        conv4_size = self._get_output_size(conv3_size, self.conv4.padding, self.conv4.dilation, self.conv4.stride, self.conv4.kernel_size)
        
        self.conv5 = torch.nn.Conv3d(in_channels=384, out_channels=256, kernel_size=3, padding=(3,1,1))
        conv5_size = self._get_output_size(conv4_size, self.conv5.padding, self.conv5.dilation, self.conv5.stride, self.conv5.kernel_size)
        pool3_size = self._get_output_size(conv5_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.linear1 = torch.nn.Linear(in_features=self.conv5.out_channels*pool3_size[0]*pool3_size[1]*pool3_size[2], out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = torch.nn.Linear(in_features=4096, out_features=output_size)
        
        self.dropout = torch.nn.Dropout(p=0.5)
        
        self.softmax = torch.nn.Softmax(dim=1)
    
    def _get_output_size(self: 'AlexNet3D', input_size: tuple, padding: int, dilation: int, stride: int, kernel_size: int) -> tuple[int, int, int]:
        """Funtion that calculate the output convolution size

        Args:
            input_size (tuple): Size of the input as [dilation, height, width]
            padding (int/tuple): Padding size
            dilation (int/tuple): dilation size
            stride (int/tuple): Stride count
            kernel_size (int/tuple): Size of the Kernel

        Returns:
            tuple[int,int,int]: Output convolution size of Dialation, Height and Width respectively
        """
        
        output = []
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, kernel_size, kernel_size)
            #padding = (padding, padding, padding)
            dilation = (dilation, dilation, dilation)
            stride = (stride, stride, stride)
        
        for idx in range(3):
            output.append(self._get_conv3d_output_size(input_size[idx], padding[idx], dilation[idx],
                                                           stride[idx], kernel_size[idx]))
        return int(output[0]), int(output[1]), int(output[2])
        
    
    def _get_conv3d_output_size(self: 'AlexNet3D', input_size: int, padding: int, dilation: int, stride: int, kernel_size: int) -> int:
        """Funtion that calculate the output convolution 3d size

        Args:
            input_size (int): Size of the input
            padding (int): Padding size
            dilation (int): dilation size
            stride (int): Stride count
            kernel_size (int): Size of the Kernel

        Returns:
            int: Output convolution size.
        """
        return np.floor(((input_size+ (2* padding) - (dilation *(kernel_size-1)) - 1)/stride)+1)
        
    
    def forward(self: 'AlexNet3D', x: torch.Tensor) -> torch.Tensor:
        """Function that performs the forward pass of the Neural Network

        Args:
            x (torch.Tensor): Input Tensor that carries that information about a batch of images

        Returns:
            torch.Tensor: Output tensor that carries the predicted probability of each class.
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool(x)
        
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool(x)
        
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.linear3(x)
        x = self.softmax(x)
        
        return x


class VGG3D(torch.nn.Module):
    
    def __init__(self: 'VGG3D', input_size: tuple, output_size: int)-> None:
        """Function to initate the model layers

        Args:
            input_size (tuple): Input size of the video as [dilation, height, width]
            
            output_size (int): Total number of output classes
        """
        super(VGG3D,self).__init__()
        
        self.max_pool = torch.nn.MaxPool3d(kernel_size=3, padding=(1,1,1),stride=2)
        
        self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=64, padding=(3,0,0), kernel_size=3)
        conv1_size = self._get_output_size(input_size, self.conv1.padding, self.conv1.dilation, self.conv1.stride, self.conv1.kernel_size)
        
        self.conv2 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=(3,2,2))
        conv2_size = self._get_output_size(conv1_size, self.conv2.padding, self.conv2.dilation, self.conv2.stride, self.conv2.kernel_size)
        pool1_size = self._get_output_size(conv2_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=(3,1,1))
        conv3_size = self._get_output_size(pool1_size, self.conv3.padding, self.conv3.dilation, self.conv3.stride, self.conv3.kernel_size)
        
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=(3,1,1))
        conv4_size = self._get_output_size(conv3_size, self.conv4.padding, self.conv4.dilation, self.conv4.stride, self.conv4.kernel_size)
        pool2_size = self._get_output_size(conv4_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.conv5 = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=(3,1,1))
        conv5_size = self._get_output_size(pool2_size, self.conv5.padding, self.conv5.dilation, self.conv5.stride, self.conv5.kernel_size)
        
        self.conv6 = torch.nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=(3,1,1))
        conv6_size = self._get_output_size(conv5_size, self.conv6.padding, self.conv6.dilation, self.conv6.stride, self.conv6.kernel_size)
        pool3_size = self._get_output_size(conv6_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.conv7 = torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=(3,4,4))
        conv7_size = self._get_output_size(pool3_size, self.conv7.padding, self.conv7.dilation, self.conv7.stride, self.conv7.kernel_size)
        
        self.conv8 = torch.nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=(3,1,1))
        conv8_size = self._get_output_size(conv7_size, self.conv8.padding, self.conv8.dilation, self.conv8.stride, self.conv8.kernel_size)
        pool4_size = self._get_output_size(conv8_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.conv9 = torch.nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=(3,4,4))
        conv9_size = self._get_output_size(pool4_size, self.conv9.padding, self.conv9.dilation, self.conv9.stride, self.conv9.kernel_size)
        
        self.conv10 = torch.nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=(3,1,1))
        conv10_size = self._get_output_size(conv9_size, self.conv10.padding, self.conv10.dilation, self.conv10.stride, self.conv10.kernel_size)
        pool5_size = self._get_output_size(conv10_size, self.max_pool.padding, self.max_pool.dilation, self.max_pool.stride, self.max_pool.kernel_size)
        
        self.linear1 = torch.nn.Linear(in_features=self.conv10.out_channels*pool5_size[0]*pool5_size[1]*pool5_size[2], out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = torch.nn.Linear(in_features=4096, out_features=output_size)
        
        self.softmax = torch.nn.Softmax(dim=1)
    
    def _get_output_size(self: 'VGG3D', input_size: tuple, padding: int, dilation: int, stride: int, kernel_size: int) -> tuple[int, int, int]:
        """Funtion that calculate the output convolution size

        Args:
            input_size (tuple): Size of the input as [dilation, height, width]
            padding (int/tuple): Padding size
            dilation (int/tuple): dilation size
            stride (int/tuple): Stride count
            kernel_size (int/tuple): Size of the Kernel

        Returns:
            tuple[int,int,int]: Output convolution size of Dialation, Height and Width respectively
        """
        
        output = []
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, kernel_size, kernel_size)
            #padding = (padding, padding, padding)
            dilation = (dilation, dilation, dilation)
            stride = (stride, stride, stride)
        
        for idx in range(3):
            output.append(self._get_conv3d_output_size(input_size[idx], padding[idx], dilation[idx],
                                                           stride[idx], kernel_size[idx]))
        return int(output[0]), int(output[1]), int(output[2])
        
    
    def _get_conv3d_output_size(self: 'VGG3D', input_size: int, padding: int, dilation: int, stride: int, kernel_size: int) -> int:
        """Funtion that calculate the output convolution 3d size

        Args:
            input_size (int): Size of the input
            padding (int): Padding size
            dilation (int): dilation size
            stride (int): Stride count
            kernel_size (int): Size of the Kernel

        Returns:
            int: Output convolution size.
        """
        return np.floor(((input_size+ (2* padding) - (dilation *(kernel_size-1)) - 1)/stride)+1)
        
    
    def forward(self: 'VGG3D', x: torch.Tensor) -> torch.Tensor:
        """Function that performs the forward pass of the Neural Network

        Args:
            x (torch.Tensor): Input Tensor that carries that information about a batch of images

        Returns:
            torch.Tensor: Output tensor that carries the predicted probability of each class.
        """
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool(x)
        
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.max_pool(x)
        
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.max_pool(x)
        
        x = torch.nn.functional.relu(self.conv7(x))
        x = torch.nn.functional.relu(self.conv8(x))
        x = self.max_pool(x)
        
        x = torch.nn.functional.relu(self.conv9(x))
        x = torch.nn.functional.relu(self.conv10(x))
        x = self.max_pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        
        return x