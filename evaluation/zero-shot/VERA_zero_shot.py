import os

import pandas as pd
import torch
import transformers
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import trange

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    torch.cuda.set_device(0)

tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera')
model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera').to(device)
model.D = model.shared.embedding_dim
linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
model.eval()
t = model.shared.weight[32097, 0].item()  # temperature for calibration

for split in ['tst']:
    data = pd.read_csv('../data/test.csv', index_col=None)
    # metaphysical_event_data = pd.concat([metaphysical_event_data, pd.read_csv('../metaphysical_event_data/dev.csv', index_col=None)]).reset_index(drop=True)

    for id in trange(len(data), desc="Predicting with VERA"):
        prompt = "If {}, then {}".format(data.loc[id, 'event_after_transition'], data.loc[id, 'inference'])

        answer = data.loc[id, 'label']

        input_ids = tokenizer.batch_encode_plus([prompt], return_tensors='pt', padding='longest',
                                                truncation='longest_first', max_length=128).input_ids.to(device)
        with torch.no_grad():
            output = model(input_ids)
            last_hidden_state = output.last_hidden_state
            hidden = last_hidden_state[0, -1, :]
            logit = linear(hidden).squeeze(-1).cpu()
            logit_calibrated = logit / t
            score_calibrated = logit_calibrated.sigmoid()
            # score_calibrated is Vera's final output plausibility score

        data.loc[id, 'generation'] = score_calibrated.item()
        data.loc[id, 'prediction_label'] = 1 if score_calibrated.item() >= 0.5 else 0

    # print accuracy, auc, macrof1
    accuracy = (data['prediction_label'] == data['label']).mean()
    auc = roc_auc_score(data['label'], data['generation'])
    f1 = f1_score(data['label'], data['prediction_label'], average='macro')
    print(f'Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Macro-F1: {f1:.4f}')
    # metaphysical_event_data.to_csv('../metaphysical_event_data/test_predictions.csv', index=False)
    # print('Predictions saved to ./VERA_zeroshot_test_predictions.csv')
