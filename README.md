# CS5260 Assignment 6

## Step 1: Initialization

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
git clone https://github.com/hpcaitech/ColossalAI.git
cd ./ColossalAI/examples/language/gpt
pip install -r requirements.txt
```

## Step 2: Use GeminiDPP/ZeRO + Tensor Parallelism

cd gemini

bash run_gemini.sh
