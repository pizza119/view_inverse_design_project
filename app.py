import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 0. 설정 및 정규화 값
# ==========================================
MEAN_THICKNESS = np.array([50.0195, 50.12645, 50.055504, 50.020386, 50.059242, 50.0466, 50.054993, 50.047863])
STD_THICKNESS = np.array([12.729691, 12.730785, 12.685574, 12.686402, 12.647134, 12.705547, 12.759413, 12.76598])
DEVICE = torch.device('cpu')

# ==========================================
# 1. 모델 클래스 정의
# ==========================================
class InverseNet_PaperSpec(nn.Module):
    def __init__(self, input_dim=201, output_dim=8):
        super(InverseNet_PaperSpec, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
            nn.Linear(1000, output_dim)
        )
    def forward(self, x): return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=201):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, output_dim)
        )
    def forward(self, x): return self.model(x)

class TandemNet(nn.Module):
    def __init__(self, inverse_model, forward_model):
        super(TandemNet, self).__init__()
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        for param in self.forward_model.parameters(): 
            param.requires_grad = False     
    def forward(self, spectrum):
        pred_thick = self.inverse_model(spectrum)
        recon_spec = self.forward_model(pred_thick)
        return pred_thick, recon_spec

# ==========================================
# 2. 모델 로드 함수
# ==========================================
@st.cache_resource
def load_models():
    f_model = MLP().to(DEVICE)
    i_model = InverseNet_PaperSpec().to(DEVICE)
    t_model = TandemNet(i_model, f_model).to(DEVICE)
    
    path = 'tandem_model_change1.pth' 
    
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        t_model.load_state_dict(checkpoint)
    except:
        try:
            # GitHub 배포 환경용 경로
            checkpoint = torch.load('tandem_model_change1.pth', map_location=DEVICE)
            t_model.load_state_dict(checkpoint)
        except:
            st.error("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            return None
    
    t_model.eval()
    return t_model

# ==========================================
# 3. 최적화 알고리즘
# ==========================================
def run_neural_adjoint(target_spec, forward_model, steps=200, lr=0.05):
    batch_size = 100 
    target_batch = target_spec.repeat(batch_size, 1)
    rand_x = torch.randn(batch_size, 8).to(DEVICE).requires_grad_(True)
    optimizer = optim.Adam([rand_x], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        pred_spec = forward_model(rand_x)
        loss = nn.functional.mse_loss(pred_spec, target_batch)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_preds = forward_model(rand_x)
        losses = torch.mean((final_preds - target_batch)**2, dim=1)
        best_idx = torch.argmin(losses)
        
    return rand_x[best_idx].unsqueeze(0).detach()

def run_hybrid(target_spec, tandem_model, steps=50, lr=0.01):
    with torch.no_grad():
        init_x, _ = tandem_model(target_spec)
    opt_x = init_x.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([opt_x], lr=lr)
    forward_model = tandem_model.forward_model
    for _ in range(steps):
        optimizer.zero_grad()
        pred_spec = forward_model(opt_x)
        loss = nn.functional.mse_loss(pred_spec, target_spec)
        loss.backward()
        optimizer.step()
    return opt_x.detach()

# ==========================================
# 4. 표 하이라이트 함수
# ==========================================
def highlight_best_model(row):
    target = row['Target']
    diffs = {
        'Tandem': abs(row['Tandem'] - target),
        'Adjoint': abs(row['Adjoint'] - target),
        'Hybrid': abs(row['Hybrid'] - target)
    }
    best_model = min(diffs, key=diffs.get)
    styles = []
    for col in row.index:
        if col == best_model:
            styles.append('background-color: #D4EDDA; color: #155724; font-weight: bold')
        else:
            styles.append('')
    return styles

# ==========================================
# 5. UI 및 로직
# ==========================================
st.set_page_config(layout="wide", page_title="AI Nano-Optics Lab")

def randomize_callback():
    new_vals = np.random.uniform(30, 70, 8)
    for i in range(8):
        st.session_state[f"slider_{i}"] = float(new_vals[i])

# --- 사이드바 ---
with st.sidebar:
    st.header("🎛️ 구조 설정")
    st.button("🎲 두께 랜덤 설정 (Randomize)", on_click=randomize_callback)
    st.divider()
    sliders = []
    for i in range(8):
        if f"slider_{i}" not in st.session_state:
            st.session_state[f"slider_{i}"] = 50.0
        val = st.slider(f"Layer {i+1} (nm)", 30.0, 70.0, key=f"slider_{i}")
        sliders.append(val)

# --- 메인 화면 ---
st.title("🧪 AI Nano-Photonic Inverse Design")
st.markdown("왼쪽 슬라이더로 **목표 구조**를 설정하면, AI가 **스펙트럼만 보고 구조를 역추적**합니다.")

model = load_models()

if model is not None:
    # 1. 정답 설정
    true_thickness_nm = np.array(sliders)
    true_thick_norm = (true_thickness_nm - MEAN_THICKNESS) / STD_THICKNESS
    true_tensor = torch.FloatTensor(true_thick_norm).unsqueeze(0).to(DEVICE)
    
    # 2. 정답 스펙트럼
    with torch.no_grad():
        target_spec_norm = model.forward_model(true_tensor)
    
    # 3. 예측 수행
    with torch.no_grad():
        pred_1_norm, _ = model(target_spec_norm) 
    pred_2_norm = run_neural_adjoint(target_spec_norm, model.forward_model)
    pred_3_norm = run_hybrid(target_spec_norm, model)
    
    # 4. 결과 복원
    def denorm(val_norm):
        val = (val_norm.cpu().numpy().flatten() * STD_THICKNESS) + MEAN_THICKNESS
        return np.clip(val, 30, 70)
        
    pred_1 = denorm(pred_1_norm)
    pred_2 = denorm(pred_2_norm)
    pred_3 = denorm(pred_3_norm)
    
    # 5. 스펙트럼 재검증
    with torch.no_grad():
        spec_1 = model.forward_model(pred_1_norm).cpu().numpy().flatten()
        spec_2 = model.forward_model(pred_2_norm).cpu().numpy().flatten()
        spec_3 = model.forward_model(pred_3_norm).cpu().numpy().flatten()
    target_spec_real = target_spec_norm.cpu().numpy().flatten()

    # --- 시각화 ---
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📊 성능 비교 (Layer-wise)")
        
        mse_1 = np.mean((pred_1 - true_thickness_nm)**2)
        mse_2 = np.mean((pred_2 - true_thickness_nm)**2)
        mse_3 = np.mean((pred_3 - true_thickness_nm)**2)
        
        scores = [mse_1, mse_2, mse_3]
        best_idx = np.argmin(scores)
        deltas = [None, None, None]
        deltas[best_idx] = "Best 👑"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("1. Tandem", f"{mse_1:.2f}", delta=deltas[0], delta_color="inverse")
        m2.metric("2. Adjoint", f"{mse_2:.2f}", delta=deltas[1], delta_color="inverse")
        m3.metric("3. Hybrid", f"{mse_3:.2f}", delta=deltas[2], delta_color="inverse")
        
        st.markdown("---")
        
        df = pd.DataFrame({
            "Layer": [f"L{i+1}" for i in range(8)],
            "Target": true_thickness_nm,
            "Tandem": pred_1,
            "Adjoint": pred_2,
            "Hybrid": pred_3
        })
        
        st.dataframe(
            df.style.format("{:.1f}", subset=["Target", "Tandem", "Adjoint", "Hybrid"])
              .apply(highlight_best_model, axis=1), 
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("📈 스펙트럼 비교")
        wavelengths = np.linspace(400, 800, 201)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=target_spec_real,
            name='Goal (Ground Truth)',
            line=dict(color='black', width=4),
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spec_1,
            name='1. Tandem',
            line=dict(color='blue', width=2),
            opacity=0.6 
        ))
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spec_2,
            name='2. Adjoint',
            line=dict(color='green', width=2),
            opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spec_3,
            name='3. Hybrid',
            line=dict(color='red', width=3),
            opacity=1.0
        ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)", yaxis_title="Reflectance",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3D 관련 코드는 모두 삭제되었습니다! (속도 최적화 완료)