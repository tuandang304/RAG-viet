"""Prompt templates for 4 noise types — Vietnamese query perturbation.

Each noise type has:
  - A system prompt describing the persona / task
  - Few-shot examples (3–5 pairs) to guide the LLM
  - A user template with a {query} placeholder

Design principles (from Ke_Hoach_Mo_Rong_Train_VLQA_LLM_Noise.md):
  - Sinh riêng từng loại, không trộn lẫn
  - Giữ nguyên hoàn toàn ý nghĩa và đáp án liên quan
  - Chỉ trả về câu hỏi đã viết lại, không giải thích
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoisePrompt:
    """A prompt template for one noise type."""
    noise_type: str
    description: str
    system: str
    few_shot_examples: list[tuple[str, str]]  # [(original, noisy), ...]
    user_template: str  # must contain {query}

    def format_user(self, query: str) -> str:
        """Build the full user message with few-shot examples + query."""
        parts = []
        for orig, noisy in self.few_shot_examples:
            parts.append(f"Ví dụ:\nGốc: {orig}\nViết lại: {noisy}")
        parts.append(f"\nCâu hỏi cần viết lại: {query}")
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 1. MẤT DẤU TỰ NHIÊN (missing_tone)
# ═══════════════════════════════════════════════════════════════════════
MISSING_TONE = NoisePrompt(
    noise_type="missing_tone",
    description="Thiếu một phần dấu thanh, sai dấu ngẫu nhiên — không xóa sạch 100%",
    system=(
        "Bạn là người Việt gõ câu hỏi pháp luật trên điện thoại, gõ vội nên "
        "thường thiếu dấu thanh ở một số từ (nhưng KHÔNG phải tất cả). "
        "Viết lại câu hỏi theo phong cách đó. "
        "Giữ nguyên hoàn toàn ý nghĩa — không thêm/bớt thông tin. "
        "Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm. /no_think"
    ),
    few_shot_examples=[
        (
            "Thủ tục ly hôn đơn phương cần những giấy tờ gì?",
            "Thu tuc ly hôn đơn phuong can nhung giay to gi?"
        ),
        (
            "Quy định về thời gian thử việc đối với lao động phổ thông là bao lâu?",
            "Quy dinh ve thoi gian thu viec doi voi lao dong pho thong la bao lau?"
        ),
        (
            "Người lao động có được đơn phương chấm dứt hợp đồng lao động không?",
            "Nguoi lao đong co duoc don phuong cham dut hop dong lao dong khong?"
        ),
        (
            "Mức phạt khi vi phạm giao thông đối với xe máy không có bằng lái?",
            "Muc phat khi vi pham giao thong đoi voi xe may khong co bang lai?"
        ),
        (
            "Điều kiện để được hưởng bảo hiểm thất nghiệp theo quy định mới nhất?",
            "Dieu kien đe duoc huong bao hiem that nghiep theo quy dinh moi nhat?"
        ),
    ],
    user_template="{query}",
)


# ═══════════════════════════════════════════════════════════════════════
# 2. LỖI GÕ TELEX / VNI (typo_telex)
# ═══════════════════════════════════════════════════════════════════════
TYPO_TELEX = NoisePrompt(
    noise_type="typo_telex",
    description="Lỗi gõ Telex/VNI — thiếu phím cuối, nhầm tổ hợp, gõ dở dang",
    system=(
        "Bạn là người Việt dùng bộ gõ Telex trên máy tính, hay gõ nhanh nên "
        "thường bị lỗi: thiếu phím cuối (ví dụ 'aw' thành chữ ă nhưng quên dấu), "
        "gõ nhầm tổ hợp, hoặc để nguyên ký tự Telex chưa chuyển đổi. "
        "Viết lại câu hỏi với một số lỗi gõ Telex tự nhiên. "
        "Giữ nguyên ý nghĩa — không thêm/bớt thông tin. "
        "Chỉ trả về câu hỏi đã viết lại, không giải thích. /no_think"
    ),
    few_shot_examples=[
        (
            "Thủ tục đăng ký kết hôn cần những giấy tờ gì?",
            "Thủ tucj ddawng ky ket hôn cần nhunwgx giấy tờ gì?"
        ),
        (
            "Quyền lợi của người lao động khi bị sa thải trái pháp luật?",
            "Quyền loiij của nguoif lao doongj khi bij sa thải tráii phápp luaajt?"
        ),
        (
            "Đồng sở hữu có quyền bán nhà đất khi chưa được sự đồng ý?",
            "Ddoongf sở hữu có quyeenf bán nhà đất khi chuaw dduocj sự ddoongf y?"
        ),
        (
            "Hợp đồng lao động có hiệu lực khi nào?",
            "Howpj ddoongf lao dộng có hieuj lucj khi naof?"
        ),
    ],
    user_template="{query}",
)


# ═══════════════════════════════════════════════════════════════════════
# 3. VIẾT TẮT / VĂN NÓI (informal)
# ═══════════════════════════════════════════════════════════════════════
INFORMAL = NoisePrompt(
    noise_type="informal",
    description="Viết tắt, văn nói, teen code — 'k' = 'không', bỏ dấu câu",
    system=(
        "Bạn là người Việt nhắn tin hỏi luật sư qua Zalo/Messenger, "
        "hay viết tắt kiểu teen: 'k' = 'không', 'dc' = 'được', 'bt' = 'biết', "
        "'j' = 'gì', 'ng' = 'người', 'mk' = 'mình', 'trc' = 'trước', "
        "'đc' = 'được', 'ko' = 'không'. Ít dùng dấu câu, viết liền. "
        "Viết lại câu hỏi theo phong cách đó. "
        "Giữ nguyên ý nghĩa — không thêm/bớt thông tin. "
        "Chỉ trả về câu hỏi đã viết lại, không giải thích. /no_think"
    ),
    few_shot_examples=[
        (
            "Tôi muốn hỏi về thủ tục ly hôn đơn phương như thế nào?",
            "e muốn hỏi thủ tục ly hôn đơn phương ntn ạ"
        ),
        (
            "Người lao động có quyền đơn phương chấm dứt hợp đồng không?",
            "ng lao động có dc đơn phương chấm dứt hợp đồng ko ạ"
        ),
        (
            "Quy định về bảo hiểm xã hội đối với lao động tự do là gì?",
            "quy định bhxh vs lao động tự do là j vậy"
        ),
        (
            "Mức phạt vi phạm giao thông khi không có giấy phép lái xe?",
            "mức phạt vi phạm gt khi k có bằng lái là bao nhiu"
        ),
        (
            "Điều kiện để được nhận trợ cấp thất nghiệp theo quy định mới?",
            "đk đc nhận trợ cấp thất nghiệp theo qđ mới là j ạ"
        ),
    ],
    user_template="{query}",
)


# ═══════════════════════════════════════════════════════════════════════
# 4. CODE-SWITCHING (code_switch)
# ═══════════════════════════════════════════════════════════════════════
CODE_SWITCH = NoisePrompt(
    noise_type="code_switch",
    description="Chèn thuật ngữ tiếng Anh tự nhiên vào câu hỏi pháp lý",
    system=(
        "Bạn là người Việt trẻ, hay dùng tiếng Anh xen lẫn tiếng Việt khi "
        "nói về pháp luật (code-switching). Ví dụ dùng 'company' thay 'công ty', "
        "'contract' thay 'hợp đồng', 'tax' thay 'thuế', 'deadline' thay 'hạn chót'. "
        "Viết lại câu hỏi bằng cách thay một số từ khóa pháp lý bằng tiếng Anh, "
        "nhưng vẫn giữ cấu trúc tiếng Việt tự nhiên. "
        "Giữ nguyên ý nghĩa — không thêm/bớt thông tin. "
        "Chỉ trả về câu hỏi đã viết lại, không giải thích. /no_think"
    ),
    few_shot_examples=[
        (
            "Thủ tục đăng ký thành lập công ty trách nhiệm hữu hạn?",
            "Thủ tục register thành lập company trách nhiệm hữu hạn?"
        ),
        (
            "Quy định về thuế thu nhập cá nhân khi bán bất động sản?",
            "Quy định về tax thu nhập cá nhân khi bán real estate?"
        ),
        (
            "Người lao động có quyền yêu cầu bồi thường khi bị sa thải?",
            "Employee có quyền yêu cầu compensation khi bị sa thải?"
        ),
        (
            "Hợp đồng lao động phải có những điều khoản bắt buộc nào?",
            "Labor contract phải có những điều khoản bắt buộc nào?"
        ),
    ],
    user_template="{query}",
)


# ── Registry ────────────────────────────────────────────────────────────
NOISE_TYPES: dict[str, NoisePrompt] = {
    "missing_tone": MISSING_TONE,
    "typo_telex": TYPO_TELEX,
    "informal": INFORMAL,
    "code_switch": CODE_SWITCH,
}

ALL_NOISE_TYPE_IDS = list(NOISE_TYPES.keys())
