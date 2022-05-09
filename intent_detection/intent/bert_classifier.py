import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel

from intent_detection import DIR_DATA, DIR_MODELS

np.random.seed(42)
use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()

model_name = "bert-base-uncased"
# model_name = 'roberta-large'
tokenizer = BertTokenizer.from_pretrained(model_name)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.labels = y
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=50,
                truncation=True,
                return_tensors="pt",
            )
            for text in x
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, output_dim, m_name=None, dropout=0.1, use_cuda=False):

        super(BertClassifier, self).__init__()

        self.model_name = m_name or "bert-base-uncased"
        self.bert = BertModel.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()
        self.path = DIR_MODELS / self.model_name
        self.use_cuda = use_cuda and torch.cuda.is_available()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output1 = self.linear1(dropout_output)
        return linear_output1

    def fit(self, train_x, train_y, x_val=None, y_val=None, **kwargs):
        batch_size = kwargs.get("batch_size", 128)
        n_iter = kwargs.get("n_iter", 5000)
        learning_rate = kwargs.get("learning_rate", 0.00005)
        weight_decay = kwargs.get("weight_decay", 0.000000)
        epochs = kwargs.get("epochs") or int(n_iter / (len(train_x) / batch_size))
        verbose = kwargs.get("verbose", False)
        dataset_name = kwargs.get("dataset_name", "")
        output_path = f"{self.path}_{dataset_name}"

        print(self.use_cuda)
        trn = Dataset(train_x, train_y)

        train_dataloader = torch.utils.data.DataLoader(
            trn, batch_size=batch_size, shuffle=True
        )
        # val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )
        # device = torch.device("cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), weight_decay=weight_decay, lr=learning_rate)

        if self.use_cuda:
            criterion = criterion.cuda()

        prev_acc = -1
        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            iteration = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input["attention_mask"].to(device)
                input_id = train_input["input_ids"].squeeze(1).to(device)

                optimizer.zero_grad()
                output = self.forward(input_id, mask)
                batch_loss = criterion(output, train_label)
                batch_loss.backward()
                optimizer.step()

                total_loss_train += batch_loss.item()
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                if iteration % 10 == 0 and verbose:
                    print(f"Iteration: {iteration}. Loss: {batch_loss.item()}.")

                iteration += 1

            if x_val:
                acc = self.evaluate(x_val, y_val)
                print(f"Testing acc: {acc:.3f}. Prev acc: {prev_acc:.3f}")

                if acc > prev_acc:
                    torch.save(self.state_dict(), output_path)  # Save model
                    prev_acc = acc
                else:
                    print("early stopping")
                    break
            else:
                acc = -1

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_y): .3f} \
                    # | Training accuracy: {total_acc_train / len(train_y): .3f}"
            )

    def evaluate(self, test_x, test_y):
        tst = Dataset(test_x, test_y)
        test_dataloader = torch.utils.data.DataLoader(tst, batch_size=8)

        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )

        total_acc_test = 0
        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                output = self.forward(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc

        return total_acc_test / len(test_y)

    def predict(self, test_x, test_y):
        tst = Dataset(test_x, test_y)
        test_dataloader = torch.utils.data.DataLoader(tst, batch_size=8)

        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )

        total_acc_test = 0
        pred = []
        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)

                output = self.forward(input_id, mask)
                # print(output.argmax(dim=1), output.argmax(dim=1).shape)
                pred.extend(output.argmax(dim=1).tolist())

        return pred


class BERT:
    def __init__(self, name="bert-base-uncased"):
        self.model_name = name
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def predict(self, x_old, **kwargs):
        vec = torch.zeros(len(x_old), 768)
        for i, x in enumerate(x_old):
            if i % 100 == 0:
                print(f"i: {i}. {(i / len(x_old)) * 100}% done")
            bert_input = self.tokenizer(
                x,
                padding="max_length",
                max_length=50,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(
                input_ids=bert_input["input_ids"],
                attention_mask=bert_input["attention_mask"],
                return_dict=False,
            )
            last_hidden_states = outputs[0][0][
                0
            ]  # The last hidden-state is the first element of the output tuple
            vec[i, :] = torch.tensor(last_hidden_states.tolist())
        return vec
