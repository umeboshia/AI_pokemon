# tenjusai-2022
Pokemon generator

by https://github.com/minimaxir/ai-generated-pokemon-rudalle

streamlitでwebアプリから入力プロンプトを取得

-> ruDALL-Eで画像を生成

-> 作成した画像をslack webhookでslackチャンネルに送信

-> 保存した画像をiphoneでシールにする(これは係員が頑張る)

```
# at shell in system
PORT=9000
IMG=tenjusai-2022
CONT=tenjusai-2022
git clone git@github.com:umeboshia/tenjusai-2022.git
cd tenjusai-2022/env
docker build . -t $IMG
cd ..
docker run -t -d --name $CONT -v $PWD:/workspace -v /{root}:/{root} --gpus all -p $PORT:$PORT $IMG
docker exec -it $CONT bash
```

before run the main script, you must set the following variables at src/main.py

- server_address
- root
- slack_url

```
# at shell in container
PORT=9000
streamlit run --server.port $PORT src/main.py
```
