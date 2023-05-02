from transformers import PreTrainedTokenizerFast, AutoTokenizer

context = ["phạm văn đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là thủ tướng đầu tiên của nước cộng hòa xã hội "
           "chủ nghĩa việt nam từ năm 1976 (từ năm 1981 gọi là chủ tịch hội đồng bộ trưởng) cho đến khi nghỉ hưu năm "
           "1987. trước đó ông từng giữ chức vụ thủ tướng chính phủ việt nam dân chủ cộng hòa từ năm 1955 đến năm "
           "1976. ông là vị thủ tướng việt nam tại vị lâu nhất (1955–1987). ông là học trò, cộng sự của chủ tịch hồ "
           "chí minh. ông có tên gọi thân mật là tô, đây từng là bí danh của ông. ông còn có tên gọi là lâm bá kiệt "
           "khi làm phó chủ nhiệm cơ quan biện sự xứ tại quế lâm (chủ nhiệm là hồ học lãm)."]
questions = ["Kinh đô ánh sáng nổi tiếng về lĩnh vực gì?"]
answers = ["thủ tướng chính phủ việt nam dân chủ cộng hòa"]
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
        print(start_index)
        print(end_index)
        print(offset_mapping)
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
