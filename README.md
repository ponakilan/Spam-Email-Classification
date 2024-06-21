## About
This project aims to classify spam emails with accelerated inference on x86-64 machines using OneDNN Graph Fusion.
This project utilizes a Long Short-Term Memory (LSTM) Neural Network to classify spam emails. 
The model is trained on the [Spam Email Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) and achieves an accuracy of 98.18% on the validation set.
A Kaggle version of this project can be found [here](https://www.kaggle.com/code/ponakilan/email-spam-classification).

## Getting Started
1. Clone the repository and change your current directory
    ```shell
    git clone https://github.com/ponakilan/Spam-Email-Classification.git
    cd Spam-Email-Classification
    ```

2. Install the requirements
    ```shell
    pip3 install -r requirements.txt
    ```

3. Start the Streamlit server
    ```shell
    streamlit run server.py
    ```
   
## Training
To train this model on your custom dataset, you can leverage the training loop provided in [train.py](https://github.com/ponakilan/Spam-Email-Classification/classifier/train.py).
The implementation includes features such as Early Stopping and Learning Rate Decay. 
Early Stopping prevents overfitting by halting training when the model's performance stops improving, and its parameters (`patience` and `min_delta`) can be adjusted in [train.py](https://github.com/ponakilan/Spam-Email-Classification/classifier/train.py).
Learning Rate Decay gradually reduces the learning rate during training to fine-tune the model, and its `gamma` parameter can also be modified in the same file.

#### Example
```python
import torch
from torch.utils.data import DataLoader

from classifier.train import train
from classifier.model import Classifier
from classifier.dataset import TextDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TextDataset(...)
valid_dataset = TextDataset(...)
train_dataloader = DataLoader(train_dataset, ...)
valid_dataloader = DataLoader(valid_dataset, ...)

model = Classifier(len(train_dataset.tokens_to_idx) + 1, ...)
history = train(
   model=model,
   train_dataloader=train_dataloader,
   valid_dataloader=valid_dataloader,
   epochs=10,
   device=device,
   show_plot=True
)
```
