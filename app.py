import os
from flask import Flask, request, redirect, url_for
from flask import jsonify
from werkzeug.utils import secure_filename

from models.encoder import EncoderSkipThought
from models.classification_models import MultimodalAttentionRNN

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import transforms
from torchvision import models

from PIL import Image

import torch.backends.cudnn as cudnn
import pickle
cudnn.benchmark = True

import spacy

nlp = spacy.load('en')
transform = transforms.Compose([
    transforms.Scale((448,448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

with open("data/vocabs.pkl", 'rb') as f:
    vocabs = pickle.load(f)

question_vocab = vocabs["question"]
ans_vocab      = vocabs["answer"]
ans_type_vocab = vocabs["ans_type"]

# Build the models
#encoder = EncoderCNN(512,models.inception_v3(pretrained=True))
encoder =  nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-2]) 
netR = EncoderSkipThought(question_vocab)
netM = MultimodalAttentionRNN(ans_vocab)


netR.load_state_dict(torch.load("data/netR.pkl", map_location=lambda storage, loc: storage))
netM.load_state_dict(torch.load("data/netM.pkl", map_location=lambda storage, loc: storage))

if torch.cuda.is_available():
    encoder = encoder.cuda()
    netM    = netM.cuda()
    netR    = netR.cuda()


netM.eval()
netR.eval()
encoder.eval()



UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

word_lookup = {"colour": "color"}
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image = Image.open(file)
            image = Variable(transform(image).unsqueeze(0), volatile=True)

            query  = request.form.get('question').lower()
            tokens = [w.text for w in nlp(query)]
            print(tokens)
            q_idx = []
            for item in tokens:
                if item in question_vocab.word2idx.keys():
                    q_idx.append(question_vocab(item))
                else:
                    if item in word_lookup.keys():
                        q_idx.append(question_vocab(word_lookup[item]))

            captions = [torch.Tensor(q_idx)]

            # Merge captions (from tuple of 1D tensor to 2D tensor).
            lengths = [len(cap) for cap in captions]
            question = torch.zeros(len(captions), max(lengths)).long()
            for i, cap in enumerate(captions):
                end = lengths[i]
                question[i, :end] = cap[:end]

            question = Variable(question, volatile=True)
            if torch.cuda.is_available():
                image   = image.cuda()
                question = question.cuda()

            visual_features = encoder(image)
            text_features, text_all_output   = netR(question, lengths)
            out = netM(visual_features, text_features, text_all_output, lengths)

            predict = torch.max(out,1)[1].data.cpu().numpy()[0][0]
            answer = ans_vocab.idx2word[predict]

            return jsonify({"status":200, "response": answer})
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type="text" name=question>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
if __name__ == "__main__":
    app.run(host="0.0.0.0")
    #app.debug = True