import torch
import whisper

from ml.HMC.utils import load_hmc_model
from ml.constants import HMC_PATH
from ml.imagebind.models import imagebind_model
from ml.summarizer.main import load_summary_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

whisper = whisper.load_model("tiny").to(device)

imagebind = imagebind_model.imagebind_huge(pretrained=True)
imagebind.eval()
imagebind.to(device)

summary_tokenizer, summary_model = load_summary_model(device)

hmc, level1_mlb, level2_mlb, level3_mlb = load_hmc_model(HMC_PATH, device)
