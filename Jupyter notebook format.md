{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Adversarial Attack Implementation and Analysis\n",
    "*Developer: Adarsh Anand*\n",
    "\n",
    "This notebook implements adversarial attacks on deep learning models, focusing on image classification tasks using the CIFAR-10 dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup and Dependencies\n",
    "First, let's import all required libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from foolbox import PyTorchModel\n",
    "from foolbox.attacks import FGSM, PGD\n",
    "import seaborn as sns\n",
    "from tqdm.notebook import tqdm\n",
    "import torch.optim as optim\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import os\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Configuration\n",
    "Setting up configuration parameters for reproducibility and experiment settings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class Config:\n",
    "    SEED = 42\n",
    "    BATCH_SIZE = 64\n",
    "    IMAGE_SIZE = 32\n",
    "    EPOCHS = 10\n",
    "    LEARNING_RATE = 0.001\n",
    "    EPSILON = 8/255\n",
    "    RESULTS_DIR = 'results'\n",
    "    \n",
    "    @staticmethod\n",
    "    def setup():\n",
    "        torch.manual_seed(Config.SEED)\n",
    "        np.random.seed(Config.SEED)\n",
    "        torch.backends.cudnn.deterministic = True\n",
    "        if not os.path.exists(Config.RESULTS_DIR):\n",
    "            os.makedirs(Config.RESULTS_DIR)\n",
    "            \n",
    "Config.setup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading\n",
    "Implementing data loading functionality for CIFAR-10 dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class DataManager:\n",
    "    def __init__(self, batch_size=Config.BATCH_SIZE):\n",
    "        self.batch_size = batch_size\n",
    "        self.transform = transforms.Compose([\n",
    "            transforms.ToTensor(),\n",
    "        ])\n",
    "        \n",
    "    def load_data(self):\n",
    "        trainset = torchvision.datasets.CIFAR10(\n",
    "            root='./data', train=True, download=True, transform=self.transform\n",
    "        )\n",
    "        testset = torchvision.datasets.CIFAR10(\n",
    "            root='./data', train=False, download=True, transform=self.transform\n",
    "        )\n",
    "        \n",
    "        trainloader = torch.utils.data.DataLoader(\n",
    "            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2\n",
    "        )\n",
    "        testloader = torch.utils.data.DataLoader(\n",
    "            testset, batch_size=self.batch_size, shuffle=False, num_workers=2\n",
    "        )\n",
    "        \n",
    "        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', \n",
    "                   'dog', 'frog', 'horse', 'ship', 'truck')\n",
    "        \n",
    "        return trainloader, testloader"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Attack Implementation\n",
    "Here we implement different adversarial attacks including FGSM and PGD."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class AdvancedAttackFramework:\n",
    "    def __init__(self, model, device):\n",
    "        self.model = model\n",
    "        self.device = device\n",
    "        self.fmodel = PyTorchModel(model, bounds=(0, 1))\n",
    "        \n",
    "    def generate_attack(self, images, labels, method='pgd', **params):\n",
    "        attacks = {\n",
    "            'fgsm': self._fgsm_attack,\n",
    "            'pgd': self._pgd_attack,\n",
    "        }\n",
    "        return attacks[method](images, labels, **params)\n",
    "    \n",
    "    def _fgsm_attack(self, images, labels, eps=Config.EPSILON):\n",
    "        attack = FGSM()\n",
    "        images = images.to(self.device)\n",
    "        labels = labels.to(self.device)\n",
    "        _, adv_images, success = attack(self.fmodel, images, labels, epsilons=[eps])\n",
    "        return adv_images[0]\n",
    "    \n",
    "    def _pgd_attack(self, images, labels, eps=Config.EPSILON, steps=10):\n",
    "        attack = PGD()\n",
    "        images = images.to(self.device)\n",
    "        labels = labels.to(self.device)\n",
    "        _, adv_images, success = attack(self.fmodel, images, labels, epsilons=[eps])\n",
    "        return adv_images[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Defense Implementation\n",
    "Implementation of defense mechanisms against adversarial attacks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class RobustDefenseModule:\n",
    "    def __init__(self, model, device):\n",
    "        self.model = model\n",
    "        self.device = device\n",
    "        \n",
    "    def gaussian_noise(self, x, sigma=0.1):\n",
    "        return torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)\n",
    "    \n",
    "    def adversarial_training(self, trainloader, attack_framework, epochs=Config.EPOCHS):\n",
    "        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)\n",
    "        criterion = nn.CrossEntropyLoss()\n",
    "        \n",
    "        for epoch in range(epochs):\n",
    "            running_loss = 0.0\n",
    "            for i, (images, labels) in enumerate(tqdm(trainloader)):\n",
    "                images, labels = images.to(self.device), labels.to(self.device)\n",
    "                \n",
    "                adv_images = attack_framework.generate_attack(images, labels)\n",
    "                \n",
    "                optimizer.zero_grad()\n",
    "                outputs_clean = self.model(images)\n",
    "                outputs_adv = self.model(adv_images)\n",
    "                \n",
    "                loss = criterion(outputs_clean, labels) + criterion(outputs_adv, labels)\n",
    "                loss.backward()\n",
    "                optimizer.step()\n",
    "                \n",
    "                running_loss += loss.item()\n",
    "                \n",
    "            print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysis Tools\n",
    "Tools for analyzing model robustness and visualizing results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class RobustnessAnalyzer:\n",
    "    def __init__(self, model, device):\n",
    "        self.model = model\n",
    "        self.device = device\n",
    "        self.results = {}\n",
    "        \n",
    "    def evaluate_model(self, dataloader, attack_framework=None, attack_type=None):\n",
    "        correct = 0\n",
    "        total = 0\n",
    "        self.model.eval()\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            for images, labels in tqdm(dataloader):\n",
    "                images, labels = images.to(self.device), labels.to(self.device)\n",
    "                \n",
    "                if attack_framework and attack_type:\n",
    "                    images = attack_framework.generate_attack(images, labels, method=attack_type)\n",
    "                \n",
    "                outputs = self.model(images)\n",
    "                _, predicted = torch.max(outputs.data, 1)\n",
    "                total += labels.size(0)\n",
    "                correct += (predicted == labels).sum().item()\n",
    "                \n",
    "        return 100 * correct / total"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Main Execution\n",
    "Running the complete pipeline and analyzing results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Setup device\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "# Load data\n",
    "data_manager = DataManager()\n",
    "trainloader, testloader = data_manager.load_data()\n",
    "\n",
    "# Initialize model\n",
    "model = torchvision.models.resnet18(pretrained=True)\n",
    "model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)\n",
    "model.fc = nn.Linear(model.fc.in_features, 10)\n",
    "model = model.to(device)\n",
    "\n",
    "# Setup components\n",
    "attack_framework = AdvancedAttackFramework(model, device)\n",
    "defense_module = RobustDefenseModule(model, device)\n",
    "robustness_analyzer = RobustnessAnalyzer(model, device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Baseline Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "print(\"Evaluating baseline performance...\")\n",
    "clean_acc = robustness_analyzer.evaluate_model(testloader)\n",
    "robustness_analyzer.results['Clean'] = clean_acc\n",
    "print(f\"Clean accuracy: {clean_acc:.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Attack Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "print(\"Evaluating attack performance...\")\n",
    "for attack_type in ['fgsm', 'pgd']:\n",
    "    print(f\"Testing {attack_type.upper()} attack...\")\n",
    "    acc = robustness_analyzer.evaluate_model(testloader, attack_framework, attack_type)\n",
    "    robustness_analyzer.results[attack_type.upper()] = acc\n",
    "    print(f\"{attack_type.upper()} accuracy: {acc:.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Results Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plotting attack performance comparison\n",
    "plt.figure(figsize=(10, 6))\n",
    "attacks = list(robustness_analyzer.results.keys())\n",
    "accuracies = list(robustness_analyzer.results.values())\n",
    "\n",
    "sns.barplot(x=attacks, y=accuracies)\n",
    "plt.title('Model Accuracy Under Different Attacks')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.ylim(0, 100)\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
</antArtifact