from transformers import PreTrainedTokenizerFast, AutoTokenizer
from e2eqavn.utils.preprocess import preprocess_qa_text
import re

context = ["triều đình seleukos cố gắng cải tạo lại jerusalem khi một thành phố theo văn minh hy lạp trở thành người đứng đầu sau khởi nghĩa maqabim thành công năm 168 tcn lãnh đạo bởi tu sỹ mattathias cùng với 5 người con trai của ông chống lại antiochus epiphanes , và họ thành lập vương quốc hasmoneus năm 152 tcn với jerusalem một lần nữa là kinh đô của vương quốc. vương quốc hasmoneus kéo dài trên một trăm năm , nhưng sau đó khi đế quốc la mã trở nên hùng mạnh hơn họ đưa herod lên làm vua chư hầu người do thái. vương quốc của vua herod cũng kéo dài trên một trăm năm. bị người do thái đánh bại trong cuộc khởi nghĩa do thái thứ nhất năm 70 , cuộc chiến tranh do thái - la mã đầu tiên và cuộc khởi nghĩa bar kokhba năm 135 đã đóng góp đáng kể vào số lượng và địa lý của cộng đồng do thái ở nước ngoài , do một phần lớn dân số do thái của vùng đất israel bị trục xuất rồi bị bán làm nô lệ trong toàn đế quốc la mã. kể từ đó , những người do thái đã sống trên mọi đất nước của thế giới , chủ yếu là ở châu âu và vùng trung đông mở rộng , trải qua nhiều sự ngược đãi , đàn áp , nghèo đói , và ngay cả diệt chủng ( xem : chủ nghĩa bài do thái , holocaust ) , với thỉnh thoảng một vài giai đoạn phát triển hưng thịnh về văn hóa , kinh tế , và tài sản cá nhân ở nhiều nơi khác nhau ( chẳng hạn như tây ban nha , bồ đào nha , đức , ba lan và hoa kỳ ) ."]
answers = ["tu sỹ mattathias cùng với 5 người con trai của ông"]
questions = ['từ những sai lầm tại cuộc đàm phán hiệp định geneva, trung quốc đã tận dụng chúng như thế nào?']
tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')


def get_index(context, answer, question, answer_start):
    context = preprocess_qa_text(context)
    answer = preprocess_qa_text(answer)
    print(tokenizer.tokenize(context))
    output_tokenizer_samples = tokenizer(
        context,
        question,
        return_offsets_mapping=True
    )
    cls_token_id = tokenizer.cls_token_id
    input_ids = output_tokenizer_samples['input_ids']
    cls_index = input_ids.index(cls_token_id)
    offset_mapping = output_tokenizer_samples['offset_mapping']

    idx = context.find(answer)
    print('*'*30)
    print(idx)
    if idx == -1:
        logger.info("Failed")
    else:
        while idx != -1 and idx < answer_start - 1:
            if answer_start is None:
                break
            elif answer_start is not None and idx < answer_start:
                idx = context.find(answer, idx + 1)
                break

    start_index = idx
    end_index = start_index + len(answer)
    print(start_index)
    print(end_index)
    print(offset_mapping)
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
    print(token_end_index, token_start_index)
    if token_end_index > -1 and token_start_index > -1:
        return input_ids, token_start_index, token_end_index
    return input_ids, cls_index, cls_index


input_ids, start_idx, end_idx = get_index(context[0], answers[0].lower(), questions[0], answer_start=68)
print(answers[0])
print(answers[0].lower() == tokenizer.decode(input_ids[start_idx: end_idx+1]))
