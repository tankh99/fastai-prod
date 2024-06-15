import gradio as gr
from fastai.vision.all import *

learn = load_learner('pickles/bear-classifier.pkl')

categories = ("black", "grizzly", "teddy")

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()
examples = ["examples/black.jpeg", "examples/grizzly.jpeg", "examples/teddy.jpg"]

iface = gr.Interface(fn=classify_image, inputs=image, outputs=label, title="Bear Classifier")
iface.launch(share=True)