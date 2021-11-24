# IQA-Server
Web backend for iOS VQA app

App repository [here](https://github.com/HiiYL/IQA-App)


### Installation
1. Download data files from [here](https://drive.google.com/open?id=0B0wpQxTjT1jOM3pwZ2psRU80TDQ) and place in data/ directory.
2. `conda create --name pytorch_env python=3.8`
3. `conda activate pytorch_env`
4. `conda install -c pytorch pytorch torchvision spacy nltk pandas`
5. `python -m spacy download en_core_web_sm`

If on mac, use brew to download wget which is used internally for skipthoughts
`brew install wget`

### Usage
1. ` python app.py `
