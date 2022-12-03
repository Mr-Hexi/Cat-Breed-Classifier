import torch
import torchvision
from timeit import default_timer as timer
import gradio as gr
from typing import Tuple ,Dict
from model import create_effnetb2_model
import os



with open("classes.txt") as f:
  classes= [line.rstrip() for line in f]

effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(classes))

effnetb2.load_state_dict(
    torch.load(
        f="Cat_Breed_Classifier_12_class_90_acc.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

def predict(img):
    start_time = timer()
    img = effnetb2_transforms(img).unsqueeze(0)
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    pred_labels_and_probs = {
        classes[i]: float(pred_probs[0][i]) for i in range(len(classes))
        }
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

title = "Cat Breed Classifier Demo 😼"
description = "<p style='text-align: center'>Gradio Demo for  Classifying Cat Breeds of these <a href='https://huggingface.co/'>5 different types.<a></p>"
article = "</br><p style='text-align: center'><a href='https://github.com/Mr-Hexi' target='_blank'>GitHub</a></br>![visitors](https://visitor-badge.glitch.me/badge?page_id=Hexii.Cat-Breed-Classifier)</p> "



example_list = [["examples/" + example] for example in os.listdir("examples")]

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
    )

app.launch()