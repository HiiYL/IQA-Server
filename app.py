import os
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from flask import Flask, jsonify, redirect, request, url_for
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
from werkzeug.utils import secure_filename

from models.classification_models import MultimodalAttentionRNN
from models.encoder import EncoderSkipThought

cudnn.benchmark = True

import spacy


class VQAModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.transform = transforms.Compose(
            [
                transforms.Scale((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.aliases = {"colour": "color"}

        with open("data/vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)

        self.question_vocab = vocabs["question"]
        self.ans_vocab = vocabs["answer"]
        self.ans_type_vocab = vocabs["ans_type"]

        # Build the models
        # encoder = EncoderCNN(512,models.inception_v3(pretrained=True))
        self.encoder = nn.Sequential(
            *list(models.resnet152(pretrained=True).children())[:-2]
        ).eval()

        self.netR = EncoderSkipThought(self.question_vocab).eval()
        self.netR.load_state_dict(
            torch.load("data/netR.pkl", map_location=lambda storage, loc: storage)
        )

        self.netM = MultimodalAttentionRNN(self.ans_vocab).eval()
        self.netM.load_state_dict(
            torch.load("data/netM.pkl", map_location=lambda storage, loc: storage)
        )

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.netM = self.netM.cuda()
            self.netR = self.netR.cuda()

    def sentence_to_idx(self, query):
        tokens = [w.text for w in self.nlp(query)]
        print(tokens)

        q_idx = []
        for token in tokens:
            if token in self.aliases.keys():
                token = self.aliases[token]
            if token in self.question_vocab.word2idx.keys():
                q_idx.append(self.question_vocab(token))

        return q_idx

    def preprocess_query(self, query):
        query_idx = self.sentence_to_idx(query)
        query_tensor = [torch.Tensor(query_idx)]

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in query_tensor]
        query_padded = torch.zeros(len(query_tensor), max(lengths)).long()
        for i, cap in enumerate(query_tensor):
            end = lengths[i]
            query_padded[i, :end] = cap[:end]

        query_padded = Variable(query_padded, volatile=True)

        return query_padded, lengths

    def evaluate(self, image, query):
        image = Variable(self.transform(image).unsqueeze(0), volatile=True)
        question, lengths = self.preprocess_query(query)

        if torch.cuda.is_available():
            image = image.cuda()
            question = question.cuda()

        visual_features = self.encoder(image)
        text_features, text_all_output = self.netR(question, lengths)
        out = self.netM(visual_features, text_features, text_all_output, lengths)

        predict = torch.max(out, 1)[1].data.cpu().numpy()[0][0]
        answer = self.ans_vocab.idx2word[predict]

        return answer


UPLOAD_FOLDER = "images/"
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vqa_model = VQAModel()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image = Image.open(file)

            query = request.form.get("question").lower()
            answer = vqa_model.evaluate(image, query)

            return jsonify({"status": 200, "response": answer})
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type="text" name=question>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    # app.debug = True
