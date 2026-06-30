{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOXPm5uHZfpy+3qxa+Qyi1k",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nainy-sara/convolutional-neural-networks/blob/main/cnn.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RifF-Q2DFmus"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import torch #pytorch\n",
        "import torch.nn as nn\n",
        "from skimage.data import shepp_logan_phantom\n",
        "from google.colab.patches import cv2_imshow\n",
        "import cv2"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "true_np = shepp_logan_phantom()\n",
        "noisy_np = np.random.poisson(true_np)\n",
        "\n",
        "def cv2disp(name, image, xpos, ypos) : cv2_imshow(image*1.0/(np.max(image)+1e-15)); #cv2.moveWindow(name, xpos, ypos) #cv2.moveWindow is not working in Colab\n",
        "\n",
        "nxd = true_np.shape[0]\n",
        "\n",
        "\n",
        "cv2disp('true', true_np, 0, 0)\n",
        "cv2disp('noisy', noisy_np, nxd, 0)\n",
        "\n",
        "class Convolution_NxN(nn.Module):\n",
        "    def __init__(self, kernal_size):\n",
        "        super(Convolution_NxN, self).__init__()\n",
        "        self.conv1kernal = nn.Conv2d(1, 1, kernal_size, padding=(int(kernal_size/2), int(kernal_size/2)), bias=False)\n",
        "        #self.conv1kernal.weight.data.fill_(1.0/(kernal_size*kernal_size))\n",
        "\n",
        "    def forward(self, x):\n",
        "            x = self.conv1kernal(x)\n",
        "            return x\n",
        "\n",
        "noisy_torch    = torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0)\n",
        "\n",
        "conv_fix_kernal = Convolution_NxN(31)\n",
        "\n",
        "kernal_torch = list(conv_fix_kernal.parameters())[0]\n",
        "cv2disp('kernel', cv2.resize(np.squeeze(kernal_torch.detach().numpy()), (nxd,nxd), interpolation = 0), 2*nxd, 0)\n",
        "\n",
        "conv_out_torch = conv_fix_kernal(noisy_torch)\n",
        "conv_out_np = np.squeeze(conv_out_torch.detach().numpy())\n",
        "\n",
        "cv2disp('output of conv', conv_out_np, 3*nxd, 0)\n",
        "\n",
        "con_trained_kernal = Convolution_NxN(31)\n",
        "loss_function   = nn.MSELoss()\n",
        "true_torch  = torch.from_numpy(true_np).float().unsqueeze(0).unsqueeze(0)\n",
        "optimiser   = torch.optim.Adam(con_trained_kernal.parameters(), lr=0.003)\n",
        "\n",
        "for epoch in range(500):\n",
        "    output = con_trained_kernal(noisy_torch)\n",
        "    loss = loss_function(output, true_torch)\n",
        "    loss.backward()\n",
        "    optimiser.step()\n",
        "    optimiser.zero_grad()\n",
        "\n",
        "    kernal_torch = list(con_trained_kernal.parameters())[0]\n",
        "    cv2disp('trained kernel', cv2.resize(np.squeeze(kernal_torch.detach().numpy()), (nxd,nxd), interpolation = 0), 1*nxd, 30+nxd)\n",
        "\n",
        "    cnn_output = np.squeeze(output.detach().numpy())\n",
        "    cv2.putText(cnn_output, 'Ep %d' % epoch, (2,30), 0, 1, int(np.max(cnn_output)+1), 1, cv2.LINE_AA)\n",
        "    cv2disp('output after trained conv', cnn_output, nxd*2, nxd+30)\n",
        "    #cv2.waitKey(1)\n",
        "\n",
        "\n",
        "class CNN(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(CNN,self).__init__()\n",
        "        self.CNN = nn.Sequential(\n",
        "            nn.Conv2d(1, 8, 7, padding=(3,3)), nn.PReLU(),\n",
        "            nn.Conv2d(8, 8, 7, padding=(3,3)), nn.PReLU(),\n",
        "            nn.Conv2d(8, 8, 7, padding=(3,3)), nn.PReLU(),\n",
        "            nn.Conv2d(8, 8, 7, padding=(3,3)), nn.PReLU(),\n",
        "            nn.Conv2d(8, 1, 7, padding=(3,3)), nn.PReLU(),\n",
        "        )\n",
        "\n",
        "    def forward(self,x):\n",
        "            x = self.CNN(x)\n",
        "            return x\n",
        "\n",
        "cnn_to_train = CNN()\n",
        "loss_function = nn.MSELoss()\n",
        "optimiser = torch.optim.Adam(cnn_to_train.parameters(), lr= 0.003)\n",
        "\n",
        "for epoch in range(500):\n",
        "    output = cnn_to_train(noisy_torch)\n",
        "    loss = loss_function(output, true_torch)\n",
        "    loss.backward()\n",
        "    optimiser.step()\n",
        "    optimiser.zero_grad()\n",
        "\n",
        "    cnn_output = np.squeeze(output.detach().numpy())\n",
        "    cv2.putText(cnn_output, 'Ep %d' % epoch, (2,30), 0, 1, int(np.max(cnn_output)+1), 1, cv2.LINE_AA)\n",
        "    cv2disp('output after trained conv', cnn_output, nxd*3, nxd+30)\n",
        "    cv2.waitKey(1)\n",
        "cv2.waitKey(0)"
      ],
      "metadata": {
        "id": "6sI9YT6oFueU"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}