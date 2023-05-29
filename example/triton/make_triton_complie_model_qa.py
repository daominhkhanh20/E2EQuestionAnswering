import torch.cuda

from e2eqavn.mrc import *
from e2eqavn.processor import QATextProcessor
from e2eqavn.utils.calculate import *
from e2eqavn.datasets import DataCollatorCustom
from e2eqavn.mrc import MRCQuestionAnsweringModel
from transformers import AutoTokenizer
import torch
from torch import nn


class QaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MRCQuestionAnsweringModel.from_pretrained('checkpoint')

    def forward(self, input_ids, attention_mask, words_length):
        input_feature = {'input_ids': input_ids, 'attention_mask': attention_mask, 'words_length': words_length}
        outs = self.model(**input_feature)
        return outs.start_logits, outs.end_logits, input_ids, words_length


tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')
qa_process = QATextProcessor()
question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
context1 = "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước Cộng hòa Xã hội " \
           "chủ nghĩa Việt Nam từ năm 1976 (từ năm 1981 gọi là Chủ tịch Hội đồng Bộ trưởng) cho đến khi nghỉ hưu năm " \
           "1987. Trước đó ông từng giữ chức vụ Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa từ năm 1955 đến năm " \
           "1976. Ông là vị Thủ tướng Việt Nam tại vị lâu nhất (1955–1987). Ông là học trò, cộng sự của Chủ tịch Hồ " \
           "Chí Minh. Ông có tên gọi thân mật là Tô, đây từng là bí danh của ông. Ông còn có tên gọi là Lâm Bá Kiệt " \
           "khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm (Chủ nhiệm là Hồ Học Lãm)."

question = " ".join(qa_process.string_tokenize(question)).strip()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QaModel().to(device).eval()
input_features = make_input_feature_qa(
    questions=[question],
    documents=[context1],
    tokenizer=tokenizer
)
data_collator = DataCollatorCustom(tokenizer=tokenizer)
input_features = data_collator(input_features)
for key, value in input_features.items():
    if isinstance(value, Tensor):
        input_features[key] = value.to(device)

traced_script_module = torch.jit.trace(model, (
    input_features['input_ids'].to(device),
    input_features['attention_mask'].to(device),
    input_features['words_length'].to(device),
)
                                       )
traced_script_module.save('model_compile/qa.pt')
