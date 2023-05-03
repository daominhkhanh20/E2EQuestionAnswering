from e2eqavn.utils.preprocess import *
from e2eqavn.keywords import *
from transformers import AutoTokenizer
import logging

tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')
logger = logging.getLogger(__name__)


def calculate_input_training_for_qa(example, is_document_right: bool = False):
    context = process_text(example[CONTEXT])
    question = process_text(example[QUESTION])
    answer = process_text(example[ANSWER])
    if question == 'mặt trận chính của cuộc chiến chống pháp là ở miền nào?':
        print(answer)
    answer_start = example.get(ANSWER_START, None)
    output_tokenizer_samples = tokenizer(
        context if is_document_right else question,
        question if is_document_right else context,
        return_offsets_mapping=True
    )
    cls_token_id = tokenizer.cls_token_id
    input_ids = output_tokenizer_samples[INPUT_IDS]
    mask = output_tokenizer_samples[ATTENTION_MASK]
    cls_index = input_ids.index(cls_token_id)
    offset_mapping = output_tokenizer_samples[OFFSET_MAPPING]
    data = {
        INPUT_IDS: input_ids,
        ATTENTION_MASK: mask
    }
    try:
        idx = context.find(answer)
        if idx == -1:
            logger.info("Failed")
        else:
            while idx != -1 and idx < answer_start - 1:
                if answer_start is None:
                    break
                elif answer_start is not None and idx < answer_start - 1:
                    idx = context.find(answer, idx + 1)
                    break

        start_index = idx
        end_index = start_index + len(answer)
        token_start_index, token_end_index = -1, -1
        flag_start_index, flag_end_index = False, False
        i = 0
        while not flag_start_index and i < len(offset_mapping):
            if offset_mapping[i][0] == start_index or (offset_mapping[i][0] < start_index < offset_mapping[i][1]):
                token_start_index = i
                flag_start_index = True
            i += 1

        i = len(input_ids) - 1
        while not flag_end_index and i > -1:
            if offset_mapping[i][1] == end_index or (offset_mapping[i][0] < end_index < offset_mapping[i][1]):
                token_end_index = i
                flag_end_index = True
            i -= 1
        if token_end_index > -1 and token_start_index > -1:
            data[START_IDX] = token_start_index
            data[END_IDX] = token_end_index
        else:
            data[START_IDX] = cls_index
            data[END_IDX] = cls_index
            logger.info(f"""
            Answer: '{answer}' \n
            Answer start: {answer_start}  {start_index}\n
            Question: '{question}' \n
            Not found in document context: {context}
            """)
        return data
    except Exception as e:
        logger.info(e)
        data[START_IDX] = cls_index
        data[END_IDX] = cls_index
        logger.info(f"""
        Answer: '{answer}' \n \
        Answer start: {answer_start}
        Not found in document context: {context}
        """)
        return data


example = {
    CONTEXT: "marx đã đánh giá lại mối quan hệ của mình với những người hegel trẻ , và trong hình thức một bức thư trả lời về chủ nghĩa vô thần của bauer viết về vấn đề do thái. tiểu luận này chủ yếu gồm một sự phê bình các ý tưởng hiện thời về các quyền dân sự và nhân quyền và giải phóng con người; nó cũng bao gồm nhiều luận điểm chỉ trích đạo do thái và cả thiên chúa giáo từ quan điểm giải phóng xã hội. engels , một người cộng sản nhiệt thành , đã khơi dậy sự quan tâm của marx với tình hình của giai cấp lao động và hướng sự chú ý của marx vào kinh tế. marx trở thành một người cộng sản và đã đặt ra các quan điểm của mình trong một loạt các bài viết được gọi là các bản thảo kinh tế và triết học năm 1844 , không được xuất bản cho tới tận thập niên 1930s. trong bản thảo , marx vạch ra một quan niệm nhân đạo của chủ nghĩa cộng sản , bị ảnh hưởng bởi triết lý của ludwig feuerbach và dựa trên sự đối lập giữa bản chất xa lạ của lao động dưới chủ nghĩa tư bản và một xã hội cộng sản trong đó con người được tự do phát triển bản chất của mình trong sản xuất tập thể.",
    QUESTION: "marx đã có hành động gì với những người hegel trẻ?",
    ANSWER: "ánh giá lại mối quan hệ của mình",
    ANSWER_START: 9
}
data = calculate_input_training_for_qa(example)
print(data[START_IDX])
print(data[END_IDX])
