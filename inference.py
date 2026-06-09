
# inference.py — VoIP/WhatsApp pipeline
import torch, torch.nn.functional as F
import numpy as np, librosa, json, random
from scipy.signal import butter, sosfilt

TARGET_SR = 16000
FIXED_LEN = 3 * TARGET_SR

def _bandpass_voip(audio, sr=TARGET_SR, low=50.0, high=7000.0):
    nyq = sr/2
    sos = butter(5, [low/nyq, high/nyq], btype='bandpass', output='sos')
    return sosfilt(sos, audio).astype(np.float32)

def _opus_artifacts(audio, bitrate_mode='medium'):
    import librosa
    sm = {'high':3,'medium':7,'low':15}[bitrate_mode]
    stft = librosa.stft(audio, n_fft=512, hop_length=128)
    mag, ph = np.abs(stft), np.angle(stft)
    k = np.ones(sm)/sm
    mag_s = np.apply_along_axis(lambda x: np.convolve(x,k,mode='same'), 0, mag)
    out = librosa.istft(mag_s*np.exp(1j*ph), hop_length=128)
    if len(out)<len(audio): out=np.pad(out,(0,len(audio)-len(out)))
    return out[:len(audio)].astype(np.float32)

def _pink_noise(audio, snr_db=25.0):
    sp = np.mean(audio**2)
    if sp<1e-10: return audio
    w = np.random.randn(len(audio))
    fr = np.fft.rfftfreq(len(audio)); fr[0]=1e-10
    noise = np.fft.irfft(np.fft.rfft(w)/np.sqrt(fr), n=len(audio)).astype(np.float32)
    noise = noise/(np.std(noise)+1e-10)*np.sqrt(sp/(10**(snr_db/10)))
    return (audio+noise).astype(np.float32)

def preprocess_for_inference(audio_path, bitrate_mode='medium', snr_db=25.0):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    audio = _bandpass_voip(audio)
    audio = _opus_artifacts(audio, bitrate_mode)
    audio = _pink_noise(audio, snr_db)
    peak = np.max(np.abs(audio))
    if peak > 1e-10: audio = audio/peak
    if len(audio)>=FIXED_LEN: audio=audio[:FIXED_LEN]
    else: audio=np.pad(audio,(0,FIXED_LEN-len(audio)))
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def load_model(weights_path, config_path, device='cpu'):
    from model_arch import RawNet3
    with open(config_path) as f: cfg = json.load(f)
    model = RawNet3()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    return model.eval().to(device), cfg

@torch.no_grad()
def predict(model, tensor, device='cpu', threshold=0.35):
    logits       = model(tensor.to(device))
    probs        = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    prob_spoofed = float(probs[1])
    label        = 'spoofed' if prob_spoofed >= threshold else 'bonafide'

    return {
        'label'         : label,
        'prob_bonafide' : float(probs[0]),
        'prob_spoofed'  : prob_spoofed,
        'confidence'    : float(probs[1] if label == 'spoofed' else probs[0]),
        'threshold_used': threshold,
    }
