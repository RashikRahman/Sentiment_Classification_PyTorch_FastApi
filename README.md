# Sentiment_Classification_PyTorch_FastApi

This is a binary text classification project. I used pytorch to create the model with help of tez. Then used FastApi to deploy the model. The model.bin file couldn't be uploaded due to it's huge file, but you can run the project and create your own model.

There are two ways to run this project. Either you can train the model on colab or train it locally. If you want to run it in colab then just copy past the whole code written in NLP_Model.py. Before running it you'll have to execute two commands. These are

```ruby
!pip install tez
!pip install transformers
```
Then you can train your model. And if you want to train locally then you'll have to do the followings.

```ruby
Open anaconda powershell or CMD
cd 'Path to cloned repo'
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tez
pip install transformers
python NLP_Model.py
```
Then your model will start trainign. If your GPU has lower then 8GB of VRAM then you'll run into CUDA out of memory error. To solve this open NLP_Model.py locate to **model.fit()** parameters and set train_bs=64 to train_bs=16 if you still face issue then again lower it to 8. For this case it is recommended to train in colab.


