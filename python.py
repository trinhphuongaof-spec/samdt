import streamlit as st
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai.errors import APIError
from math import log1p

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh",
    layout="wide"
)

st.markdown("<h1 style='color: green;'>AI ĐÁNH GIÁ PHƯƠNG ÁN KINH DOANH 📈</h1>", unsafe_allow_html=True)
st.caption("Ứng dụng trích xuất thông tin tài chính từ văn bản và tính toán hiệu quả đầu tư.")

# Khởi tạo session state để lưu trữ các tham số đã trích xuất
if 'params' not in st.session_state:
    st.session_state.params = None
if 'cashflow_df' not in st.session_state:
    st.session_state.cashflow_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'file_content' not in st.session_state:
    st.session_state.file_content = ""

# --- A. HÀM TRÍCH XUẤT DỮ LIỆU BẰNG AI (GOAL 1) ---
def extract_data_with_ai(prompt_content, api_key):
    """Gửi nội dung văn bản tới Gemini để trích xuất và trả về JSON."""
    
    if not api_key:
        return "Lỗi: Thiếu Khóa API 'GEMINI_API_KEY'.", None

    # Định nghĩa cấu trúc JSON đầu ra bắt buộc
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "Vốn_Dau_Tu": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (tỷ đồng, số tuyệt đối)"},
            "Dong_Doi_Du_An": {"type": "INTEGER", "description": "Dòng đời dự án (số năm)"},
            "Doanh_Thu_Nam": {"type": "NUMBER", "description": "Doanh thu hàng năm (tỷ đồng)"},
            "Chi_Phi_Nam": {"type": "NUMBER", "description": "Chi phí hàng năm (tỷ đồng)"},
            "WACC_Phan_Tram": {"type": "NUMBER", "description": "Chi phí vốn (WACC) dưới dạng phần trăm (ví dụ: 13 cho 13%)"},
            "Thue_TNDN_Phan_Tram": {"type": "NUMBER", "description": "Thuế TNDN dưới dạng phần trăm (ví dụ: 20 cho 20%)"}
        },
        "required": ["Vốn_Dau_Tu", "Dong_Doi_Du_An", "Doanh_Thu_Nam", "Chi_Phi_Nam", "WACC_Phan_Tram", "Thue_TNDN_Phan_Tram"]
    }
    
    system_prompt = (
        "Bạn là trợ lý trích xuất dữ liệu tài chính. Hãy đọc nội dung văn bản dưới đây và trích xuất sáu chỉ tiêu tài chính. "
        "Yêu cầu TẤT CẢ các giá trị phải là SỐ (không có đơn vị tiền tệ, không có ký hiệu %, chỉ dùng dấu chấm là dấu thập phân). "
        "Đưa ra kết quả dưới dạng JSON theo schema đã cung cấp. Nếu không tìm thấy một tham số, hãy trả về 0."
    )
    
    user_prompt = f"Trích xuất các thông số tài chính từ văn bản sau:\n\n{prompt_content}"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
        # Parse JSON
        params = json.loads(response.text)
        return "Trích xuất thành công!", params
    
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}", None
    except json.JSONDecodeError:
        return "Lỗi phân tích JSON: AI không trả về cấu trúc JSON hợp lệ.", None
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}", None

# --- B. HÀM TÍNH TOÁN DÒNG TIỀN VÀ CHỈ SỐ (GOAL 2 & 3) ---

def calculate_financial_metrics(params):
    """Tính toán bảng dòng tiền và các chỉ số NPV, IRR, PP, DPP."""
    
    # Lấy tham số (đảm bảo chuyển đổi phần trăm về số thập phân)
    C0 = params.get('Vốn_Dau_Tu')
    T = params.get('Dong_Doi_Du_An')
    R = params.get('Doanh_Thu_Nam')
    C = params.get('Chi_Phi_Nam')
    WACC = params.get('WACC_Phan_Tram') / 100.0 if params.get('WACC_Phan_Tram') else 0.0
    Tax = params.get('Thue_TNDN_Phan_Tram') / 100.0 if params.get('Thue_TNDN_Phan_Tram') else 0.0
    
    if T <= 0 or C0 <= 0:
        return "Dòng đời dự án và Vốn đầu tư phải lớn hơn 0.", None, None

    # 1. Tính Khấu hao (D) - Giả định khấu hao đường thẳng
    D = C0 / T
    
    # 2. Xây dựng Bảng Dòng tiền (Năm 0 -> Năm T)
    years = list(range(T + 1))
    
    # Khởi tạo các list dòng tiền
    EBT_list = [-C0] # EBT không liên quan năm 0
    Tax_list = [0.0]
    NI_list = [-C0] # NI không liên quan năm 0
    ACF_list = [-C0] # Dòng tiền chiết khấu (Chi phí)
    
    # Tính toán cho các năm 1 đến T
    for year in range(1, T + 1):
        # EBT: Lợi nhuận trước thuế
        EBT = R - C - D
        
        # Tax: Thuế TNDN (chỉ tính khi EBT > 0)
        Tax_amount = max(0, EBT) * Tax
        
        # NI: Lợi nhuận sau thuế
        NI = EBT - Tax_amount
        
        # ACF: Dòng tiền hoạt động thuần (NI + Khấu hao)
        ACF = NI + D
        
        EBT_list.append(EBT)
        Tax_list.append(Tax_amount)
        NI_list.append(NI)
        ACF_list.append(ACF)

    # Tạo DataFrame dòng tiền
    cashflow_df = pd.DataFrame({
        'Năm (t)': years,
        'Doanh thu (R)': [0.0] + [R] * T,
        'Chi phí (C)': [0.0] + [C] * T,
        'Khấu hao (D)': [0.0] + [D] * T,
        'Lợi nhuận trước thuế (EBT)': EBT_list,
        'Thuế TNDN (20%)': Tax_list,
        'Lợi nhuận sau thuế (NI)': NI_list,
        'Dòng tiền thuần (ACF)': ACF_list
    })
    
    # 3. Tính toán các Chỉ số Hiệu quả (Goals 3)
    
    # Chuyển đổi ACF_list sang mảng numpy để tính toán
    np_acf = np.array(ACF_list)
    
    # a. NPV (Giá trị hiện tại thuần)
    discount_factors = [1 / ((1 + WACC)**t) for t in years]
    NPV = np.sum(np_acf * discount_factors)
    
    # b. IRR (Tỷ suất hoàn vốn nội bộ) - Dùng np.irr
    try:
        IRR = np.irr(np_acf)
    except:
        IRR = np.nan
        
    # c. PP (Thời gian hoàn vốn) & d. DPP (Thời gian hoàn vốn có chiết khấu)
    
    cumulative_cf = np_acf.cumsum()
    cumulative_dcf = (np_acf * discount_factors).cumsum()
    
    PP = 'Không hoàn vốn'
    DPP = 'Không hoàn vốn'
    
    # Tính PP
    for i in range(1, T + 1):
        if cumulative_cf[i] >= 0:
            # Năm trước khi hoàn vốn
            year_prev = i - 1
            # Phần tiền cần bù đắp trong năm đó
            balance_needed = -cumulative_cf[i-1]
            # Dòng tiền không chiết khấu năm hoàn vốn
            cf_year = ACF_list[i]
            PP = year_prev + (balance_needed / cf_year)
            break

    # Tính DPP
    for i in range(1, T + 1):
        if cumulative_dcf[i] >= 0:
            # Năm trước khi hoàn vốn có chiết khấu
            year_prev = i - 1
            # Phần tiền cần bù đắp trong năm đó
            balance_needed = -(np_acf * discount_factors)[i-1] # Giá trị chiết khấu lũy kế năm trước đó
            # Dòng tiền chiết khấu năm hoàn vốn
            dcf_year = (np_acf * discount_factors)[i]
            DPP = year_prev + (balance_needed / dcf_year)
            break
            
    metrics = {
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP,
    }
    
    return "Tính toán thành công!", cashflow_df, metrics

# --- C. HÀM PHÂN TÍCH CHỈ SỐ BẰNG AI (GOAL 4) ---
def get_analysis_by_ai(cashflow_df, metrics, params, api_key):
    """Gửi các chỉ số và dòng tiền đến Gemini để phân tích."""
    
    if not api_key:
        return "Lỗi: Thiếu Khóa API 'GEMINI_API_KEY'."
        
    # Chuẩn bị dữ liệu đầu vào cho AI
    project_params = pd.DataFrame(params.items(), columns=['Chỉ tiêu', 'Giá trị']).to_markdown(index=False)
    cashflow_md = cashflow_df.style.format(precision=2).to_markdown()
    
    # Định dạng các chỉ số cho AI
    npv_str = f"{metrics['NPV']:,.2f} tỷ VND"
    irr_str = f"{metrics['IRR'] * 100:,.2f}%" if not np.isnan(metrics['IRR']) else "Không xác định"
    pp_str = f"{metrics['PP']:,.2f} năm" if isinstance(metrics['PP'], (int, float)) else metrics['PP']
    dpp_str = f"{metrics['DPP']:,.2f} năm" if isinstance(metrics['DPP'], (int, float)) else metrics['DPP']

    prompt = f"""
    Bạn là một chuyên gia phân tích đầu tư và thẩm định dự án.
    Hãy phân tích tính khả thi và rủi ro của dự án dựa trên các thông số và chỉ số sau:
    
    **THÔNG SỐ DỰ ÁN:**
    {project_params}
    
    **CÁC CHỈ SỐ HIỆU QUẢ DỰ ÁN:**
    - NPV (Giá trị hiện tại thuần): {npv_str}
    - IRR (Tỷ suất hoàn vốn nội bộ): {irr_str}
    - PP (Thời gian hoàn vốn): {pp_str}
    - DPP (Thời gian hoàn vốn có chiết khấu): {dpp_str}
    
    **DÒNG TIỀN DỰ ÁN (tỷ đồng):**
    {cashflow_md}
    
    **Yêu cầu phân tích:**
    1. **Đánh giá Khả thi:** Nhận xét NPV và so sánh IRR với WACC (13%).
    2. **Đánh giá Rủi ro Thanh khoản:** Nhận xét về PP và DPP.
    3. **Kết luận:** Tóm tắt và đưa ra khuyến nghị chấp thuận/từ chối hoặc cần điều chỉnh/yêu cầu thêm dữ liệu (nếu NPV < 0). Viết nhận xét chuyên nghiệp (khoảng 3 đoạn).
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# -------------------------- BẮT ĐẦU GIAO DIỆN STREAMLIT --------------------------

api_key = st.secrets.get("GEMINI_API_KEY")

# --- Mục 1: Tải File và Trích xuất Dữ liệu ---
st.header("1. Trích xuất Thông số Dự án (AI)")
col_file, col_text = st.columns([1, 2])

with col_file:
    uploaded_file = st.file_uploader(
        "Tải lên File .txt (Hoặc copy-paste nội dung Word vào bên cạnh)",
        type=['txt']
    )
    if uploaded_file is not None:
        try:
            string_data = uploaded_file.getvalue().decode("utf-8")
            st.session_state.file_content = string_data
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

with col_text:
    # Text area để người dùng có thể copy-paste trực tiếp từ Word
    st.session_state.file_content = st.text_area(
        "Nội dung văn bản (Từ Word)",
        value=st.session_state.file_content,
        height=300,
        placeholder="Dán nội dung phương án kinh doanh vào đây. Ví dụ: 'Dự án có vốn đầu tư 30 tỷ, dòng đời 10 năm. Doanh thu 5 tỷ, chi phí 2 tỷ, WACC 13%, Thuế 20%.'"
    )

if st.button("🚀 Lọc dữ liệu tài chính (Dùng AI)"):
    if not st.session_state.file_content.strip():
        st.warning("Vui lòng nhập hoặc tải lên nội dung phương án kinh doanh.")
    else:
        with st.spinner('Đang gửi nội dung và chờ Gemini trích xuất thông tin...'):
            message, params = extract_data_with_ai(st.session_state.file_content, api_key)
            
            if params:
                st.session_state.params = params
                st.success(message)
                
                # Tính toán ngay sau khi trích xuất thành công
                calc_message, df, metrics = calculate_financial_metrics(st.session_state.params)
                st.session_state.cashflow_df = df
                st.session_state.metrics = metrics
                
                # Hiển thị thông số đã trích xuất
                st.subheader("Thông số đã Trích xuất (Đơn vị: Tỷ VND trừ khi có ghi chú)")
                st.dataframe(pd.DataFrame(st.session_state.params.items(), columns=['Chỉ tiêu', 'Giá trị']).style.format(precision=2), use_container_width=True)
                
            else:
                st.error(message)

# --- Mục 2: Bảng Dòng tiền (Goal 2) ---
if st.session_state.cashflow_df is not None:
    st.markdown("---")
    st.header("2. Bảng Dòng tiền của Dự án (Tỷ VND)")
    st.dataframe(st.session_state.cashflow_df.style.format(
        {col: '{:,.2f}' for col in st.session_state.cashflow_df.columns}
    ), use_container_width=True)

    # --- Mục 3: Chỉ số Đánh giá Hiệu quả (Goal 3) ---
    st.markdown("---")
    st.header("3. Các Chỉ số Đánh giá Hiệu quả Đầu tư")
    
    if st.session_state.metrics:
        m = st.session_state.metrics
        
        # Định dạng kết quả
        npv_str = f"{m['NPV']:,.2f} tỷ VND"
        irr_str = f"{m['IRR'] * 100:,.2f}%" if not np.isnan(m['IRR']) else "Không xác định"
        pp_str = f"{m['PP']:,.2f} năm" if isinstance(m['PP'], (int, float)) else m['PP']
        dpp_str = f"{m['DPP']:,.2f} năm" if isinstance(m['DPP'], (int, float)) else m['DPP']

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("NPV (Giá trị hiện tại thuần)", npv_str)
        with col2:
            st.metric("IRR (Tỷ suất hoàn vốn nội bộ)", irr_str)
        with col3:
            st.metric("PP (Thời gian hoàn vốn)", pp_str)
        with col4:
            st.metric("DPP (Hoàn vốn có chiết khấu)", dpp_str)

    # --- Mục 4: Phân tích AI (Goal 4) ---
    st.markdown("---")
    st.header("4. Nhận xét Chuyên sâu (AI)")
    
    if st.button("💬 Yêu cầu AI Phân tích Hiệu quả Dự án"):
        if api_key:
            with st.spinner("Đang gửi chỉ số và chờ Gemini phân tích..."):
                ai_analysis = get_analysis_by_ai(
                    st.session_state.cashflow_df, 
                    st.session_state.metrics, 
                    st.session_state.params, 
                    api_key
                )
                st.info(ai_analysis)
        else:
            st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

elif st.session_state.params is not None and st.session_state.cashflow_df is None:
    st.error("Lỗi tính toán: Vui lòng kiểm tra lại các thông số trích xuất (Đặc biệt là Dòng đời dự án phải là số nguyên dương).")
