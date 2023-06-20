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
        self.model = MRCQuestionAnsweringModel.from_pretrained('khanhbk20/mrc_testing')

    def forward(self, input_ids, attention_mask, align_matrix):
        input_feature = {'input_ids': input_ids, 'attention_mask': attention_mask}
        context_embedding = self.model.roberta(**input_feature, return_dict=True)[0]
        context_embedding_align = torch.bmm(align_matrix, context_embedding)
        logits = self.model.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous().to(torch.float32)
        end_logits = end_logits.squeeze(-1).contiguous().to(torch.float32)
        return start_logits, end_logits, input_ids, align_matrix


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
data_collator = DataCollatorCustom(tokenizer=tokenizer, mode_triton=True)
input_features = data_collator(input_features)
for key, value in input_features.items():
    if isinstance(value, Tensor):
        input_features[key] = value.to(device)

traced_script_module = torch.jit.trace(
    model, (
        input_features['input_ids'].to(device),
        input_features['attention_mask'].to(device),
        input_features['align_matrix'].to(device),
    ), strict=False
)
traced_script_module.save('model_compile/qa/model.pt')
