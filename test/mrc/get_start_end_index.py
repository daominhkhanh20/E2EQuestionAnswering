from transformers import PreTrainedTokenizerFast, AutoTokenizer

context = ["ông việt phương, nguyên thư ký của thủ tướng phạm văn đồng, trong buổi họp báo giới thiệu sách của các "
           "nhà ngoại giao, đã tiết lộ rằng khi đàm phán hiệp định geneva (1954), do đoàn việt nam không có điện đài "
           "nên bộ trưởng ngoại giao lúc đó là phạm văn đồng đã mắc một sai lầm khi nhờ trung quốc chuyển các bức "
           "điện về nước, do vậy trung quốc biết hết các sách lược của việt nam và sử dụng chúng để ép việt nam ký "
           "hiệp định theo lợi ích của trung quốc. trong đàm phán phạm văn đồng sử dụng phiên dịch trung quốc nên nội "
           "dung liên lạc giữa đoàn đàm phán và trung ương, trung quốc đều biết trước và tìm cách ngăn chặn. ông phạm "
           "văn đồng sau này cũng thừa nhận là đoàn việt nam khi đó quá tin đoàn trung quốc. tại hội nghị ấy, "
           "ông đồng chỉ chủ yếu tiếp xúc với đoàn liên xô và đoàn trung quốc, trong khi anh là đồng chủ tịch, "
           "quan điểm lại khác với pháp, nhưng ông lại không tranh thủ, không hề tiếp xúc với phái đoàn anh."]
answers = ["sử dụng chúng để ép việt nam ký hiệp định theo lợi ích của trung quốc"]
questions = ['từ những sai lầm tại cuộc đàm phán hiệp định geneva, trung quốc đã tận dụng chúng như thế nào?']
tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')


def get_index(context, answer, question):
    output_tokenizer_samples = tokenizer(
        context,
        question,
        return_offsets_mapping=True
    )
    cls_token_id = tokenizer.cls_token_id
    input_ids = output_tokenizer_samples['input_ids']
    cls_index = input_ids.index(cls_token_id)
    offset_mapping = output_tokenizer_samples['offset_mapping']

    try:
        start_index = context.index(answer)
        end_index = start_index + len(answer)
        token_start_index, token_end_index = -1, -1
        flag_start_index, flag_end_index = False, False
        i = 0
        while not flag_start_index and i < len(offset_mapping):
            if offset_mapping[i][0] == start_index:
                token_start_index = i
                flag_start_index = True
            i += 1

        i = len(input_ids) - 1
        while not flag_end_index and i > -1:
            if offset_mapping[i][1] == end_index:
                token_end_index = i
                flag_end_index = True
            i -= 1
        print(token_end_index, token_start_index)
        if token_end_index > -1 and token_start_index > -1:
            return input_ids, token_start_index, token_end_index
        return input_ids, cls_index, cls_index
    except:
        return input_ids, cls_index, cls_index


input_ids, start_idx, end_idx = get_index(context[0], answers[0], questions[0])
print(answers[0])
print(answers[0] == tokenizer.decode(input_ids[start_idx: end_idx+1]))
