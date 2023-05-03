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
        return_offsets_mapping=True,
        max_length=512,
        truncation=True
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
    CONTEXT: "năm 420 , lưu dụ tiếm vị triều đông tấn , lập ra triều lưu tống. đến năm 422 , quân bắc ngụy dưới sự chỉ đạo của ngụy minh nguyên đế đã vượt hoàng hà , sau lần cuộc chiến này , lưu tống bị mất vùng đất từ hồ lục , hạng thành trở lên phía bắc. nhân lúc bắc ngụy phải đối phó với nhu nhiên phía bắc , năm 429 , lưu tống văn đế đòi bắc ngụy trả đất hà nam song ngụy thái vũ đế không chịu. năm 430 , lưu tống văn đế sai đáo ngạn chi mang quân bắc phạt , quân bắc ngụy ít nên chủ động rút lui song sau đó đã phản công , lấy lại được lạc dương và hổ lao. tháng 2 năm 431 , quân ngụy giao tranh với quân lưu tống do đàn đạo tế chỉ huy lên bắc cứu hoạt đài ( nay là huyện hoạt ) , hai bên đánh nhau 30 trận , đều bị tổn thất nặng , cuối cùng đàn đạo tế phải đưa quân lưu tống rút lui. bắc ngụy thống nhất hoàn toàn miền bắc trung quốc vào năm 439 , cùng với các triều đại ở phương nam mở ra thời nam bắc triều trong lịch sử trung quốc. năm 450 , bắc ngụy thái vũ đế điều 10 vạn quân vây đánh huyền hồ ( 懸瓠 , nay thuộc trú mã điếm ) trong 42 ngày , quân lưu tống ở các thành xung quanh đều sợ thế bắc ngụy và bỏ thành rút chạy. sau khi đánh bại quân lưu tống được cử đến cứu viện , thái vũ đế lại đem quân bắc ngụy nam tiến , kết quả chiếm được vùng hoài bắc. năm 493 , bắc ngụy hiếu văn đế đã cho mang đến hà dương ( hà nam ) 2 triệu con ngựa , tổ chức quân túc vệ 15 vạn thường trú tại lạc dương , dời hộ tịch toàn bộ người vùng đại tới lạc dương. hành động dời kinh đô đến lạc dương là để hiển thị bắc ngụy là chính quyền chính thống của trung quốc và có lợi cho việc hấp thụ mau lẹ văn hóa hán. sau khi hiếu vũ đế sang quan trung với tập đoàn quân phiệt quan lũng của vũ văn thái , cao hoan đã lập nguyên thiện kiến làm vua mới , tức là đông ngụy hiếu tĩnh đế. do thấy lạc dương gần họ vũ văn , ông thiên đô về nghiệp",
    QUESTION: "điều gì được thể hiện thông qua việc lạc dương trở thành kinh đô?",
    ANSWER: "bắc ngụy là chính quyền chính thống của trung quốc và có lợi cho việc hấp thụ mau lẹ văn hóa hán",
    ANSWER_START: 1462
}
data = calculate_input_training_for_qa(example)
print(data[START_IDX])
print(data[END_IDX])
