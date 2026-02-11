import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 모델 클래스 정의 (반드시 있어야 로드 가능)
# ==========================================

# 맨 처음 y값을 줄 때 예측 x를 내뱉는 MLP(학습시켜야 함)
class InverseNet_PaperSpec(nn.Module):
    def __init__(self, input_dim=201, output_dim=8):
        super(InverseNet_PaperSpec, self).__init__()
        # 논문 스펙: 은닉층 4개, 뉴런 1000개
        # Batch Norm과 Dropout은 최신 트렌드를 반영해 추가
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Layer 2
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Layer 3
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Layer 4
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),

            # Output Layer (두께 8개 출력)
            nn.Linear(1000, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 2. 탠덤 네트워크 정의 (Inverse + Frozen Forward)
class TandemNet(nn.Module):
    def __init__(self, inverse_model, forward_model):
        super(TandemNet, self).__init__()
        self.inverse_model = inverse_model
        self.forward_model = forward_model

        # Forward Model은 학습하지 않도록 얼리기
        self.forward_model.eval()
        for param in self.forward_model.parameters():
            param.requires_grad = False

    def forward(self, spectrum):
        predicted_thickness_norm = self.inverse_model(spectrum) # y -> x_p(정규화 o)
        reconstructed_spectrum = self.forward_model(predicted_thickness_norm) # x_p(정규화 o)-> y_p

        return predicted_thickness_norm, reconstructed_spectrum # x_p, y_p 출력

    def train(self, mode=True):
      super(TandemNet, self).train(mode) # 일단 전체를 모드에 맞게 변경
      self.forward_model.eval()          # 그 다음 Forward만 강제로 eval로 고정
      return self
    

# Forward Model (시뮬레이터 대체용)
# 완전열결 MLP 모델 구현
class MLP(nn.Module):
    def __init__(self, input_dim = 8, output_dim =201, hidden_dim_1=250, hidden_dim_2=250, hidden_dim_3=250, hidden_dim_4=250):
        super(MLP, self).__init__() # 부모 클래스 __init__ 실행
        self.model = nn.Sequential(
            # 1번째 층 8 -> 250
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            # 2번째 층 250 -> 250
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            # 3번째 층 250 -> 250
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            # 4번째 층 250 -> 250
            nn.Linear(hidden_dim_3, hidden_dim_4),
            nn.ReLU(),
            # 5번째 층 250 -> 201
            nn.Linear(hidden_dim_4, output_dim)
        )

        # 가중치 초기화 (논문: Normal dist, mean=0, std=0.1)
        self._initialize_weights() # 정규분포로 가중치, bios 초기화

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)


# ==========================================
# 2. 모델 로드 함수 (캐싱으로 속도 향상)
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cpu') # 서버에는 GPU가 없을 수 있으니 CPU로
    
    # 깡통 모델 생성
    f_model = MLP().to(device)
    i_model = InverseNet_PaperSpec().to(device)
    t_model = TandemNet(i_model, f_model).to(device)
    
    # 가중치 로드 (파일 이름이 정확해야 함!)
    # 만약 Tandem 안에 Forward가 포함되어 저장됐다면 tandem만 로드해도 됨
    try:
        t_model.load_state_dict(torch.load('tandem_model_change1.pth', map_location=device))
    except:
        st.error("모델 파일을 찾을 수 없습니다. GitHub에 .pth 파일을 올렸는지 확인하세요.")
    
    t_model.eval()
    return t_model, device

# ==========================================
# 3. 데이터 정규화 값 (하드코딩 추천)
# ==========================================
# Colab에서 print(train_dataset.mean), print(train_dataset.std) 해서 나온 값을 적으세요.
MEAN_THICKNESS = np.array([50.0195, 50.12645, 50.055504, 50.020386, 50.059242, 50.0466, 50.054993, 50.047863])  
STD_THICKNESS = np.array([12.729691, 12.730785, 12.685574, 12.686402, 12.647134, 12.705547, 12.759413, 12.76598])  

# ==========================================
# 4. 메인 화면 (UI)
# ==========================================
st.title("🌈 AI Nano-Photonic Inverse Design")
st.markdown("원하는 **스펙트럼(반사율 패턴)**을 입력하면, AI가 그 구조를 만드는 **나노 박막 두께**를 찾아줍니다.")

# 사이드바 입력
st.sidebar.header("Target Spectrum 설정")
target_wl = st.sidebar.slider("중심 파장 (Center Wavelength)", 400, 800, 600)
width = st.sidebar.slider("반사폭 (Width)", 10, 100, 30)

# 실행 버튼
if st.button("AI 설계 시작 (Design)"):
    model, device = load_models()
    
    # 1. 가상의 목표 스펙트럼 생성 (Gaussian 형태)
    wavelengths = np.linspace(400, 800, 201)
    target_spectrum = np.exp(-((wavelengths - target_wl)**2) / (2 * width**2))
    
    # 2. AI 예측 (Tandem Network)
    # Numpy -> Tensor 변환
    input_tensor = torch.FloatTensor(target_spectrum).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Tandem 모델이 두께와 예상 스펙트럼을 동시에 뱉어줌
        pred_thickness_norm, recon_spectrum = model(input_tensor)
        
    # 3. 결과 변환 (정규화 해제)
    pred_thickness_norm = pred_thickness_norm.cpu().numpy().flatten()
    final_thickness = (pred_thickness_norm * STD_THICKNESS) + MEAN_THICKNESS
    
    # 범위 강제 (30~70nm) - 보기 좋게
    final_thickness = np.clip(final_thickness, 30, 70)
    
    # 4. 결과 출력
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ 설계 완료!")
        st.write("AI가 제안한 8층 두께 (nm):")
        st.dataframe(final_thickness)
        
    with col2:
        # 5. 그래프 그리기
        st.write("📊 스펙트럼 비교 검증")
        fig, ax = plt.subplots()
        ax.plot(wavelengths, target_spectrum, 'k--', label='Target (Goal)', linewidth=2)
        ax.plot(wavelengths, recon_spectrum.cpu().numpy().flatten(), 'r-', label='AI Result', linewidth=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Response")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)