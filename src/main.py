from lib2to3.pgen2 import token
from rudalle.pipelines import generate_images, show
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.utils import seed_everything
from huggingface_hub import hf_hub_url, cached_download
import os
import torch
import shutil
from pathlib import Path
import streamlit as st
import requests
import json
from googletrans import Translator
import os
import datetime

# before run this script, you should set the following variables
server_address = "ip adress to server" #example: "xxx.xxx.xxx.xxx""
root = "project root at server" #example: "/home/tenjusai-2022/"
slack_url = "slack_webhook_url" #exapmle: "https://hooks.slack.com/services/xxxxxxxxx/xxxxxxxxx/xxxxxxxxxxxxxxxxxxxxxxxx"


gen_configs = [
    (2048, 0.995),
    (1536, 0.99),
    (1024, 0.99),
    (1024, 0.98),
    (512, 0.97),
    (384, 0.96),
    (256, 0.95),
    (128, 0.95),
]
model_filename = "pytorch_model.bin"
device = "cuda"

# @title Download Models and Install/Load Packages (may take a few minutes)
# st session states
if "config_file_url" in st.session_state:
    config_file_url = st.session_state.config_file_url
else:
    config_file_url = hf_hub_url(repo_id="minimaxir/ai-generated-pokemon-rudalle", filename=model_filename)
    cached_download(config_file_url, cache_dir=".cache", force_filename=model_filename)
    st.session_state["config_file_url"] = config_file_url

if "model" in st.session_state:
    model = st.session_state.model
else:
    model = get_rudalle_model("Malevich", pretrained=False, fp16=True, device=device)
    model.load_state_dict(torch.load(Path(".cache") / model_filename, map_location="cpu"))
    st.session_state["model"] = model
if "vae" in st.session_state:
    vae = st.session_state.vae
else:
    vae = get_vae().to(device)
    st.session_state["vae"] = vae
if "tokenizer" in st.session_state:
    tokenizer = st.session_state.tokenizer
else:
    tokenizer = get_tokenizer()
    st.session_state["tokenizer"] = tokenizer

if "print_choices" not in st.session_state:
    st.session_state["print_choices"] = []

if "pil_images" not in st.session_state:
    st.session_state["pil_images"] = []

if "names" not in st.session_state:
    st.session_state["names"] = []


# contents
st.title("ポケモンを作ろう")
st.header("設定")
st.text("係（かかり）の人（ひと）が操作（そうさ）します")

images_per_row = st.slider("横（よこ）に何個（なんこ）を作（つく）る？", min_value=1, max_value=9, value=1, step=1)
num_rows = st.slider("縦（たて）に何個（なんこ）を作（つく）る？", min_value=1, max_value=9, value=1, step=1)
seed = st.number_input("シード", min_value=0, max_value=100000, value=0, step=1)

st.header("日本語（にほんご）で文（ぶん）を入力（にゅうりょく）してください．")
text_ja = st.text_input("入力：")

#generate images
generate_run = False
if st.button("スタート"):
    generate_run = True
    tr = Translator()
    text = tr.translate(text=text_ja, src="ja", dest="ru").text
    st.session_state["print_choices"] = []
    st.session_state["names"] = []
    st.session_state["pil_images"] = []
    cols = st.columns(images_per_row)
    dt = datetime.datetime.now(datetime.timezone.utc)
    dir = f"{dt.month}月{dt.day}日_{dt.hour+9}時_{dt.minute}分_{text_ja}"
    os.makedirs(Path(root) / dir, exist_ok=True)

    gen_configs = gen_configs[0:num_rows]

    pil_images = []
    names = []
    scores = []

    img_count = 0
    for top_k, top_p in gen_configs:
        _pil_images, _scores = generate_images(
            text, tokenizer, model, vae, top_k=top_k, images_num=images_per_row, top_p=top_p, seed=seed
        )
        _names = []
        for images in _pil_images:
            name = f"{dir}/{img_count:03d}.png"
            _names.append(name)
            images.save(Path(root) / name)
            st.session_state["print_choices"].append(name)
            img_count += 1
        # show(_pil_images, images_per_row, size=int(28 / 2))
        for i, img in enumerate(_pil_images):
            with cols[i]:
                st.text(_names[i])
                st.image(img)
        pil_images.append(_pil_images)
        names.append(_names)
    st.session_state["pil_images"] = pil_images
    st.session_state["names"] = names
    # show([pil_image for pil_image in pil_images], images_per_row, size=56)


# display images
pil_images = st.session_state["pil_images"]
if pil_images and not generate_run:
    cols = st.columns(len(pil_images[0]))
    for row, rown in zip(pil_images, st.session_state["names"]):
        for i, (img, nm) in enumerate(zip(row, rown)):
            with cols[i]:
                st.text(nm)
                st.image(img)


#save images
print_img = st.selectbox("印刷用画像", st.session_state["print_choices"])
if st.button("送信"):
    # send URL to slack
    post_json = {
        "text": "http://{}/public/tenjusai-2022/{}".format(server_address, print_img),
    }
    requests.post(slack_url, data=json.dumps(post_json))
