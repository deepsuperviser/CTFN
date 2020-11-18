import torch
import numpy
from modules.LightWeightTrans import EmotionClassifier
from modules.DualEncoder import DoubleTrans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def cal_test_result(predicted, test_label, test_mask):
    true_label = []
    predicted_label = []
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            if test_mask[i, j] == 1:
                true_label.append(test_label[i, j])
                predicted_label.append(numpy.argmax(predicted[i, j]))

    acc = round(accuracy_score(true_label, predicted_label), 4)
    f1 = round(f1_score(true_label, predicted_label, average='weighted'), 4)
    return acc, f1

def specific_modal_fusion(true_data, fake_data, mid_data):
    alphas = torch.sum(torch.abs(true_data - fake_data), (1, 2))
    alphas = torch.div(alphas, torch.sum(alphas)).unsqueeze(-1).unsqueeze(-1)
    return torch.mul(alphas, mid_data[-1])


def initiate(stop_l, config, train_loader, valid_loader, test_loader, mask, gpu_device, metrics_file, set_str):
    sa_model = EmotionClassifier(config)
    sa_model.to(gpu_device)
    optimizer = torch.optim.Adam(sa_model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    a2t_model = DoubleTrans(config, main_dim=config.audio_dim, middle_dim=config.text_dim,
                            d_model=(config.a_d_model, config.t_d_model),
                            num_head=(config.a_heads, config.t_heads),
                            num_layer=(config.a_num_layer, config.t_num_layer),
                            dim_forward=(config.a_dim_forward, config.t_dim_forward),
                            device=gpu_device, p=0.5)

    def train():
        epoch_loss = 0.0
        sa_model.train()
        for data in train_loader:
            tmp_audio, tmp_text, labels = data
            tmp_audio = tmp_audio.to(gpu_device)
            tmp_text = tmp_text.to(gpu_device)
            batch_size = tmp_audio.size(0)

            a2t_model.train(tmp_audio, tmp_text)
            a2t_fake_a, a2t_fake_t, bimodal_at, bimodal_ta = a2t_model.double_fusion(tmp_audio, tmp_text,
                                                                                     need_grad=True)

            audio_fusion = specific_modal_fusion(tmp_audio, a2t_fake_a, bimodal_ta)
            text_fusion = specific_modal_fusion(tmp_text, a2t_fake_t, bimodal_at)

            outputs = sa_model(tmp_audio, tmp_text, (audio_fusion, text_fusion)).to("cpu")

            optimizer.zero_grad()

            loss = criterion(outputs.transpose(1, 2), labels)
            if numpy.isnan(loss.item()):
                print('training loss got in NaN, stop_len: ', stop_l)
                metrics_file.close()
                assert False
            loss.backward()
            optimizer.step()
            a2t_model.grad_step()
            epoch_loss += loss.item() * batch_size
        return epoch_loss

    def evaluate(load_test=False):
        sa_model.eval()
        total_loss = 0.0
        results = []
        truths = []
        loader = test_loader if load_test else valid_loader
        with torch.no_grad():
            for data in loader:
                tmp_audio, tmp_text, labels = data
                tmp_audio = tmp_audio.to(gpu_device)
                tmp_text = tmp_text.to(gpu_device)
                batch_size = tmp_audio.size(0)

                a2t_fake_a, a2t_fake_t, bimodal_at, bimodal_ta = \
                    a2t_model.double_fusion(tmp_audio, tmp_text, need_grad=False)

                audio_fusion = specific_modal_fusion(tmp_audio, a2t_fake_a, bimodal_ta)
                text_fusion = specific_modal_fusion(tmp_text, a2t_fake_t, bimodal_at)

                outputs = sa_model(tmp_audio, tmp_text, (audio_fusion, text_fusion)).to("cpu")
                total_loss += criterion(outputs.transpose(1, 2), labels).item() * batch_size
                results.append(outputs)
                truths.append(labels)
        if load_test:
            l_acc, l_f1 = cal_test_result(torch.cat(results, dim=0), torch.cat(truths, dim=0), mask[1])
        else:
            l_acc, l_f1 = cal_test_result(torch.cat(results, dim=0), torch.cat(truths, dim=0), mask[0])
        return total_loss, (l_acc, l_f1)
    
    train_loss = []
    for epoch in range(config.epochs):
        l_train_loss = train()
        train_loss.append(round(l_train_loss, 2))

        test_loss, metrics = evaluate(load_test=True)
        metrics_file.writelines(set_str + str(metrics) + '\n')
        metrics_file.flush()


