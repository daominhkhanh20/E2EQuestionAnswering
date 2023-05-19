from e2eqavn.mrc import *
from e2eqavn.processor import QATextProcessor
qa_process = QATextProcessor()
question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
context1 = "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước Cộng hòa Xã hội " \
          "chủ nghĩa Việt Nam từ năm 1976 (từ năm 1981 gọi là Chủ tịch Hội đồng Bộ trưởng) cho đến khi nghỉ hưu năm " \
          "1987. Trước đó ông từng giữ chức vụ Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa từ năm 1955 đến năm " \
          "1976. Ông là vị Thủ tướng Việt Nam tại vị lâu nhất (1955–1987). Ông là học trò, cộng sự của Chủ tịch Hồ " \
          "Chí Minh. Ông có tên gọi thân mật là Tô, đây từng là bí danh của ông. Ông còn có tên gọi là Lâm Bá Kiệt " \
          "khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm (Chủ nhiệm là Hồ Học Lãm)."
context2 = "Năm 1954, ông được giao nhiệm vụ Trưởng phái đoàn Chính phủ dự Hội nghị Genève về Đông Dương. Những đóng góp của đoàn Việt Nam do ông đứng đầu là vô cùng quan trọng, tạo ra những đột phá đưa Hội nghị tới thành công. Trải qua 8 phiên họp toàn thể và 23 phiên họp rất căng thẳng và phức tạp, với tinh thần chủ động và cố gắng của phái đoàn Việt Nam, ngày 20/7/1954, bản Hiệp định đình chỉ chiến sự ở Việt Nam, Campuchia và Lào đã được ký kết thừa nhận tôn trọng độc lập, chủ quyền, của nước Việt Nam, Lào và Campuchia."

context1 = " ".join(qa_process.string_tokenize(qa_process.strip_context(context1))).strip()
context2 = " ".join(qa_process.string_tokenize(qa_process.strip_context(context2))).strip()

question = " ".join(qa_process.string_tokenize(question)).strip()
mrc_reader = MRCReader.from_pretrained('model/qa/checkpoint-1144')
print(mrc_reader.qa_inference(question=question, documents=[context1, context2]))