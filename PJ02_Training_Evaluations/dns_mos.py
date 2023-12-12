import concurrent.futures
import pathlib
from enum import Enum
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
class Model_Type(Enum):
        is_personalized_MOS = 1,
        not_personalized_MOS = 2

class ComputeScore:
    def __init__(self, primary_model_path, model_type : Model_Type.not_personalized_MOS):
        self.onnx_sess = ort.InferenceSession(str(primary_model_path))
        self.model_type = model_type

    def _get_polyfit_val(self, sig, bak, ovr):

        if self.model_type == Model_Type.is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs

        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            oi = {'input_1': input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self._get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw)

            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        return fpath.name, np.mean(predicted_mos_ovr_seg)


# {'filename': fpath.name, 'num_hops': num_hops, 'OVRL': np.mean(predicted_mos_ovr_seg),
#                 'SIG': np.mean(predicted_mos_sig_seg), 'BAK': np.mean(predicted_mos_bak_seg)}

def main(fpath, model_type=Model_Type.not_personalized_MOS):
    primary_model_path = pathlib.Path.cwd()
    if model_type == Model_Type.is_personalized_MOS:
        primary_model_path = primary_model_path.parent.joinpath('PJ02_Training_evaluations', 'pDNSMOS',
                                                                'sig_bak_ovr.onnx')
    else:
        primary_model_path = primary_model_path.parent.joinpath('PJ02_Training_evaluations', 'DNSMOS',
                                                                'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path,model_type=model_type)



    rows = []
    clips = fpath
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(compute_score, clip, SAMPLING_RATE): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)
    rows.sort()
    return rows
        # score_path = fpath.joinpath('score_dns')
        # if not score_path.exists():
        #     score_path.mkdir(parents=True, exist_ok=True)
        # csv_path = score_path.joinpath(f'score_dns_{model.name[-4:]}.slv')
        # g = shelve.open(str(csv_path))
        # g['file_name'] = df['filename']
        # g['dns_ovrl'] = df['OVRL']
        # g.close()

#
# if __name__ == "__main__":
#     fpath = pathlib.Path(
#         "F:/Projekts/Cruze/Results/1_New_Aproach/FFT=512/Alternative CNN-RNN Modells/Model 2/2_end_to_end_descend_corrected/DNS_MOS")
#     main(fpath)
